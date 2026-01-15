print(f"Loading {__file__}...")

import copy
import os
import time as ttime
from datetime import datetime
from pathlib import Path
import pandas as pd
import warnings
import logging

import matplotlib as mpl
import nslsii
import redis
from bluesky_queueserver import is_re_worker_active, parameter_annotation_decorator
from IPython.terminal.prompts import Prompts, Token
from ophyd.signal import DEFAULT_CONNECTION_TIMEOUT, EpicsSignal, EpicsSignalBase
from redis_json_dict import RedisJSONDict
from tiled.client import from_profile, from_uri


# Start to list warnings that we don't want to see - or just hide them from users
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)  # For 53-slitscans
warnings.filterwarnings(action="ignore", message="MSG_SIZE_TOO_LARGE")  # Kafka messages
warnings.filterwarnings(action="ignore", message="About to call kickoff()")  #  Future deprecation warning for kickoff
# Removes Qt: Session management error
if "SESSION_MANAGER" in os.environ:
    del os.environ["SESSION_MANAGER"]
# Set bluesky log file locations to DEFAULT locations
#   Removes text that it is setting these locations
#   Just kidding, doesn't remove text
os.environ["BLUESKY_LOG_FILE"] = '/home/xf05id1/.cache/bluesky/log/bluesky.log'
os.environ["BLUESKY_IPYTHON_LOG_FILE"] = '/home/xf05id1/.cache/bluesky/log/bluesky_ipython.log'


def if_touch_beamline(envvar="TOUCHBEAMLINE"):
    value = os.environ.get(envvar, "false").lower()
    if value in ("", "n", "no", "f", "false", "off", "0"):
        return False
    elif value in ("y", "yes", "t", "true", "on", "1"):
        return True
    else:
        raise ValueError(f"Unknown value: {value}")


def print_now():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S.%f")


def wait_for_connection_base(self, timeout=DEFAULT_CONNECTION_TIMEOUT):
    """Wait for the underlying signals to initialize or connect"""
    if timeout is DEFAULT_CONNECTION_TIMEOUT:
        timeout = self.connection_timeout
    # print(f'{print_now()}: waiting for {self.name} to connect within {timeout:.4f} s...')
    start = ttime.time()
    try:
        self._ensure_connected(self._read_pv, timeout=timeout)
        # print(f'{print_now()}: waited for {self.name} to connect for {time.time() - start:.4f} s.')
    except TimeoutError:
        if self._destroyed:
            raise DestroyedError("Signal has been destroyed")
        raise


def wait_for_connection(self, timeout=DEFAULT_CONNECTION_TIMEOUT):
    """Wait for the underlying signals to initialize or connect"""
    if timeout is DEFAULT_CONNECTION_TIMEOUT:
        timeout = self.connection_timeout
    # print(f'{print_now()}: waiting for {self.name} to connect within {timeout:.4f} s...')
    start = ttime.time()
    self._ensure_connected(self._read_pv, self._write_pv, timeout=timeout)
    # print(f'{print_now()}: waited for {self.name} to connect for {time.time() - start:.4f} s.')


EpicsSignalBase.wait_for_connection = wait_for_connection_base
EpicsSignal.wait_for_connection = wait_for_connection
###############################################################################

if if_touch_beamline():
    # Case of real beamline:
    timeout = 2  # seconds
    going = "Going"
else:
    # Case of CI:
    timeout = 10  # seconds
    going = "NOT going"

# print(f"\nEpicsSignalBase timeout is {timeout} [seconds]. {going} to touch beamline hardware.\n")

# EpicsSignalBase.set_default_timeout(timeout=timeout, connection_timeout=timeout)  # old style
EpicsSignalBase.set_defaults(timeout=timeout, connection_timeout=timeout)  # new style

with open("/etc/bluesky/redis.secret", "r") as f:
    redis_secret = f.read().strip()
    os.environ["REDIS_PASSWORD"] = redis_secret



ip = get_ipython()
nslsii.configure_base(
    ip.user_ns,
    "srx",
    publish_documents_with_kafka=True,
    redis_url="xf05id2-srx-redis1.nsls2.bnl.gov",
    redis_port=6380,
    redis_ssl=True,
)

# This is a workaround until the flyer can be rewritten
if 'nuke_the_cache' not in RE.commands:
    async def _nuke_cache(msg, *, self = RE):
        print(f"{print_now()}: Nuking cache...")
        run_key = msg.run
        obj = msg.obj
        print(f"{print_now()}:   {run_key=}")
        print(f"{print_now()}:   {obj=}")
        if (
                current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
        ) is key_absence_sentinel:
            current_run = None
        if current_run is not None:
            print(f"{print_now()}:   {current_run._describe_collect_cache=} popping!")
            if obj in current_run._describe_collect_cache.keys():
                print("obj in keys()")
                current_run._describe_collect_cache.pop(obj)
            print(f"{print_now()}:   {current_run._describe_collect_cache=}")

    RE.register_command('nuke_the_cache', _nuke_cache)


# Hack from above for creating streams with different data shapes
# This message clears the describe cache for a specified detector
if 'clear_describe_cache' not in RE.commands:
    async def _clear_describe_cache(msg, *, self = RE):
        run_key = msg.run
        obj = msg.obj
        if (
                current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
        ) is key_absence_sentinel:
            current_run = None
        if current_run is not None:
            if obj in current_run._describe_cache.keys():
                current_run._describe_cache.pop(obj)

    RE.register_command('clear_describe_cache', _clear_describe_cache)



RE.unsubscribe(0)

# Define tiled catalog
srx_raw = from_profile("nsls2", api_key=os.environ["TILED_BLUESKY_WRITING_API_KEY_SRX"])["srx"]["raw"]

c = tiled_reading_client = from_uri(
        "https://tiled.nsls2.bnl.gov/api/v1/metadata/srx/raw",
        include_data_sources=True,
)

discard_liveplot_data = True
descriptor_uids = []

# Temporary: This is for debugging post document
log_ipy = logging.getLogger('bluesky')

def post_document(name, doc):
    if name == "start":
        doc = copy.deepcopy(doc)
        descriptor_uids.clear()

    if name == "descriptor":
        if discard_liveplot_data and doc["name"].startswith("DONOTSAVE_"):
            descriptor_uids.append(doc["uid"])
            return
    elif name == "event_page" and doc["descriptor"] in descriptor_uids:
        return
    # print(f"==================  name={name!r} doc={doc} type(doc)={type(doc)}")
    ATTEMPTS = 120
    error = None
    for attempt in range(ATTEMPTS):
        try:
            start_time = ttime.time()
            srx_raw.post_document(name, doc)
            # print(f"{name} {doc} Post dt = {ttime.time() - start_time}")
            # log_ipy.debug(f"{name} {doc} Post dt = {ttime.time() - start_time}")
        except Exception as exc:
            print(f"[{print_now()}] Document saving failure ({attempt+1}/{ATTEMPTS}):", repr(exc))
            error = exc
            # raise exc
        else:
            break
        ttime.sleep(2)
    else:
        # Out of attempts
        raise error


RE.subscribe(post_document)

ip.log.setLevel("WARNING")

nslsii.configure_olog(ip.user_ns)

# Custom Matplotlib configs:
mpl.rcParams["axes.grid"] = True  # grid always on


# Comment it out to enable BEC table:
bec.disable_table()


# Disable BestEffortCallback to plot ring current
bec.disable_plots()

# RE.md = RedisJSONDict(redis.Redis("info.srx.nsls2.bnl.gov"), prefix="")

# Optional: set any metadata that rarely changes.
# RE.md["beamline_id"] = "SRX"
# RE.md["md_version"] = "1.1"


class SRXPrompt(Prompts):
    def in_prompt_tokens(self, cli=None):
        return [
            (
                Token.Prompt,
                f"BlueSky@SRX | Proposal #{RE.md.get('proposal', {}).get('proposal_id', 'N/A')} [",
            ),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, "]: "),
        ]


ip.prompts = SRXPrompt(ip)

# from bluesky.utils import ts_msg_hook
# RE.msg_hook = ts_msg_hook

# The following plan stubs are automatically imported in global namespace by 'nslsii.configure_base',
# but have signatures that are not compatible with the Queue Server. They should not exist in the global
# namespace, but can be accessed as 'bps.one_1d_step' etc. from other plans.
del one_1d_step, one_nd_step, one_shot
