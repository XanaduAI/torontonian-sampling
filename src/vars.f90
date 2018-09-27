! Copyright 2018 Xanadu Quantum Technologies Inc.

! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at

!     http://www.apache.org/licenses/LICENSE-2.0

! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
module vars
    use kinds
    implicit none

    integer(ip)         :: n, ell
    real(wp), parameter :: eps = 1.0d-6
    real(wp), parameter :: pi = acos(-1.0_wp)
end module vars
