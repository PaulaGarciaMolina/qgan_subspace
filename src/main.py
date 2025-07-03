# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
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
"""Main module for running the quantum GAN training and testing."""

from config import CFG
from tools.data.data_managers import print_and_log
from tools.training_init import (
    run_multiple_trainings_from_common_init_and_later_change,
    run_multiple_trainings_no_common_init,
    run_single_training,
)


##############################################################
# SINGLE RUN mode
##############################################################
def main():
    if CFG.run_multiple_experiments:
        if CFG.start_from_common_initial_experiment:
            run_multiple_trainings_from_common_init_and_later_change()
        else:
            run_multiple_trainings_no_common_init()
    else:
        print_and_log("Running in SINGLE RUN mode.\n", CFG.log_path)
        run_single_training()


if __name__ == "__main__":
    main()
