# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from .round import round as _round
from .round import round_out as _round_out

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def special_round(input, *, decimals=0):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ROUND")
    return _round(input, decimals=decimals)


def special_round_out(input, out, *, decimals=0):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ROUND_OUT")
    return _round_out(input, decimals=decimals, out=out)
