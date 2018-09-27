module kinds

  implicit none

  integer, parameter :: sp = selected_real_kind(5,30)
  integer, parameter :: dp = selected_real_kind(9,99)
  integer, parameter :: qp = selected_real_kind(20,199)
  integer, parameter :: wp = dp
  integer, parameter :: ip = selected_int_kind(16)

! These empty arrays are used to initialize variables to either the min or
! max possible number of kind wp or integer.
  real(wp), dimension(2:1) :: empty
  integer(ip), dimension(2:1) :: iempty
end module kinds

module vars

  use kinds

  implicit none

  integer(ip) :: n, ell

!  real(wp), dimension(:,:), allocatable :: covmat
!  real(wp), dimension(:), allocatable :: mean

  real(wp), parameter :: eps = 1.0d-6

  real(wp), parameter :: pi = acos(-1.0_wp)



!  namelist /input/ n, ell
end module vars



module structures

  use kinds

  implicit none

  type GaussianState
!    integer(ip) :: ell                           ! size of the state

    real(wp), dimension(:), allocatable :: rbar   ! Mean vector of the state

    real(wp), dimension(:,:), allocatable :: V    ! Covariance matrix of the state

    real(wp) :: C                                 ! Probability Coefficients

  end type GaussianState


  real(wp) :: time, dtime
end module structures



module TorontonianSamples
  use kinds
  use vars
  use structures
 
  implicit none

contains

  subroutine hello(k, n, kk) ! Written for compilation testing
    use kinds
    implicit none

    real(wp) :: k
    integer(ip) :: n
    real(wp), dimension(1:n), intent(out) :: kk
    real(wp), dimension(:), allocatable :: kk_tmp

    !f2py intent(in) :: k
    !f2py intent(in) :: n
    !f2py intent(out) :: kk

    allocate(kk_tmp(1:n))

    kk_tmp(:)=50.0_wp

    kk = kk_tmp
 
    deallocate(kk_tmp)

  end subroutine hello


  
  subroutine GenerateSample(covmat, mean, n_sample, seed, sample_out)
    use kinds
    use vars
    use structures

    implicit none

    integer(ip) :: n_sample
    integer :: seed_tmp
    real(wp), dimension(:, :), intent(in) :: covmat
    real(wp), dimension(:), intent(in) :: mean
    integer(ip), intent(in) :: seed
    integer(ip), dimension(:), allocatable :: sample
    integer(ip), dimension(1:n_sample), intent(out) :: sample_out

    integer(ip) :: i, j, k, ntmp, jj, NewStateSize
    real(wp) :: qq, qq1, qq2, random, TotalProb
    type(GaussianState), dimension(1) :: state
    type(GaussianState), dimension(:), allocatable :: state_tmp1, state_tmp2
 
!    real(wp), intent(out) :: sample_out
 
    !f2py intent(in) :: covmat
    !f2py intent(in) :: mean, seed, n_sample
    !f2py intent(out) :: sample_out


    seed_tmp = seed

    n = size(mean)
    ell = n/2


!    sample_out = mean
!    print*, "n=", n
  

    call random_seed(seed_tmp)
  
  
    allocate(sample(1:ell))
  

!!    open(4, file='samples.dat', status='replace', action='write')
!  
    do i=1,size(state)
      allocate(state(i)%rbar(1:n),state(i)%V(1:n,1:n))
    end do

  
    state(1)%rbar = mean
    state(1)%V = covmat


    allocate(state_tmp1(1:size(state)))
  
  
    state_tmp1 = state

    
    call compute_q(state_tmp1(1)%V, state_tmp1(1)%rbar, qq)
  
    TotalProb = qq
  
!    print*, 'First qq = '  qq

  
    state_tmp1(:)%C = 1.0_wp
  
    do i=1,ell
      
      ntmp = size(state_tmp1)
  
      call random_number(random)   
!      print*, "Random = ", random

  
      NewStateSize = size(state_tmp1(1)%rbar) - 2
  
      

      if (NewStateSize >= 2) then
        if (qq > random) then
    
          sample(i) = 0
  
          do k=1,size(state_tmp1)
            call compute_q(state_tmp1(k)%v, state_tmp1(k)%rbar, qq1)
            state_tmp1(k)%c = state_tmp1(k)%c * qq1/qq
          end do
  
          allocate(state_tmp2(1:ntmp))
      
          do jj=1,ntmp
            allocate(state_tmp2(jj)%rbar(1:NewStateSize), &
                                          state_tmp2(jj)%V(1:NewStateSize, 1:NewStateSize))
          end do
  
  
          do j=1,ntmp
            state_tmp2(j)%V = NewCovMat(state_tmp1(j)%V)
            state_tmp2(j)%rbar = NewMean(state_tmp1(j)%rbar, state_tmp1(j)%V)
            state_tmp2(j)%C = state_tmp1(j)%C
          end do
      
          deallocate(state_tmp1)
          allocate(state_tmp1(1:size(state_tmp2)))
          state_tmp1 = state_tmp2
          deallocate(state_tmp2)
  
          qq2 = 0.0_wp
          qq1 = 0.0_wp
      
  
          do k=1,size(state_tmp1)
            call compute_q(state_tmp1(k)%v, state_tmp1(k)%rbar, qq1)
            qq2 = qq2 + qq1*state_tmp1(k)%c
          end do
  
          
           
        else if (qq <= random) then
  
          sample(i) = 1
            allocate(state_tmp2(1:2*ntmp))
            do jj=1,2*ntmp
              allocate(state_tmp2(jj)%rbar(1:NewStateSize), &
                                            state_tmp2(jj)%V(1:NewStateSize, 1:NewStateSize))
            end do
    
            do j=1,ntmp
              state_tmp2(2*j-1)%V = NewCovMat(state_tmp1(j)%V)
              state_tmp2(2*j-1)%rbar = NewMean(state_tmp1(j)%rbar, state_tmp1(j)%V)
              call compute_q(state_tmp1(j)%V, state_tmp1(j)%rbar, qq1)
              state_tmp2(2*j-1)%C = - state_tmp1(j)%C * qq1/(1.0_wp-qq)
      
              state_tmp2(2*j)%V = VA(state_tmp1(j)%V)
              state_tmp2(2*j)%rbar = rA(state_tmp1(j)%rbar, state_tmp1(j)%V)
              state_tmp2(2*j)%C = state_tmp1(j)%C /(1.0_wp-qq)
            end do
      
          deallocate(state_tmp1)
          allocate(state_tmp1(1:size(state_tmp2)))
      
          state_tmp1 = state_tmp2
      
          deallocate(state_tmp2)
      
          qq2 = 0.0_wp
          qq1 = 0.0_wp
  
          if (abs(sum(state_tmp1(:)%C) - 1.0_wp) > 1e-6) then
            print*, "Error: Sum of all coefficients not 1. Sum =", sum(state_tmp1(:)%C)
            STOP
          end if
      
          do k=1,size(state_tmp1)
            call compute_q(state_tmp1(k)%V, state_tmp1(k)%rbar, qq1)
            qq2 = qq2 + qq1*state_tmp1(k)%C
          end do 
  
        end if
  
      else
        if (qq > random) then
           sample(i) = 0
        else if (qq <= random) then
           sample(i) = 1
        end if
      end if
  
!      write(4, 10), real(i,wp) , qq, real(sample(i), wp)
!      write(4, 10)
  
      qq = qq2
  
      if (qq < -eps) then
        print*, "Error: Probability is less than 0. q = ", qq
!        STOP
      else if (qq - 1.0_wp > 0.01_wp) then
        print*, "Error: Probability is less greater than 1. q = ", qq
!        STOP
!      else 
!        print*, "Probability of not clicking: q = ", qq
      end if
  
      if (qq < 0.0_wp) then
        print*, 'Warning: probability was a small negative number, q=', qq, "Set to Zero!"
        qq = 0.0_wp
      end if
  
      TotalProb = TotalProb * qq
  
    end do

!    sample_tmp = 5100.0_wp

    forall(i=1:ell) sample_out(i) = sample(ell+1-i)

!  
!    print*, "Sample: "
!    write(*, "(I1.1)"), sample
!    print*, "Total Probability of the Samples:", TotalProb
!  
!  
10 format(6es14.2e2)
20 format(12es20.2e2)
!  
!    do i=1,size(state)
!      deallocate(state(i)%rbar, state(i)%V)
!    end do
!  
!    close(4)
  end subroutine GenerateSample

  pure function VA(mat)
    use kinds
    use vars

    real(wp), dimension(:,:), intent(in) :: mat
    real(wp), dimension(:,:), allocatable :: VA

    integer(ip) :: i, nn

    nn = sqrt(real(size(mat),wp))
  
    allocate(VA(1:nn-2,1:nn-2))

    VA = mat(1:nn-2,1:nn-2)

  end function VA


  pure function rA(rbar, mat)
    use kinds
    use vars

    real(wp), dimension(:,:), intent(in) :: mat
    real(wp), dimension(:), intent(in) :: rbar
    real(wp), dimension(:), allocatable  :: rA

    integer(ip) :: i, nn

    nn = sqrt(real(size(mat),wp))

    allocate(rA(1:nn-2) )
  
    rA = rbar(1:nn-2)

  end function rA

  pure function NewMean(rbar, mat)
    use kinds
    use vars

    real(wp), dimension(:,:), intent(in) :: mat
    real(wp), dimension(:), intent(in) :: rbar
    real(wp), dimension(1:2)  :: rB
    real(wp), dimension(:), allocatable  :: rA
    real(wp), dimension(2,2) :: VB, VB_inv
    real(wp), dimension(:,:), allocatable :: VAB
    real(wp), dimension(:), allocatable :: NewMean

    real(wp) :: detVB
    integer(ip) :: i, nn, IdSize

    IdSize = 2

    nn = sqrt(real(size(mat),wp))

    allocate(VAB(1:nn-2,1:2), NewMean(1:nn-2), rA(1:nn-2) )
  
    rB = rbar(nn-1:nn)
    rA = rbar(1:nn-2)

    VAB = mat(1:nn-2,nn-1:nn)
    VB = mat(nn-1:nn,nn-1:nn) + identity(IdSize)
    detVB = VB(1,1)*VB(2,2) - VB(1,2)*VB(2,1)
    VB_inv = reshape((/VB(2,2), -VB(2,1), -VB(1,2), VB(1,1)/), (/2,2/))/detVB


    NewMean(:) = rA - matmul(VAB, matmul(VB_inv,rB))

  end function NewMean

  pure function NewCovMat(mat)
    use kinds
    use vars

    real(wp), dimension(:,:), intent(in) :: mat
    real(wp), dimension(2,2) :: VB, VB_inv
    real(wp), dimension(:,:), allocatable :: VAB, VAB_T, VA
    real(wp), dimension(:, :), allocatable :: NewCovMat
!    real(wp), dimension(1:nn-2, 1:nn-2) :: NewCovMat
    real(wp) :: detVB

    integer(ip) :: i, nn, IdSize

    IdSize = 2

    nn = sqrt(real(size(mat),wp))
  
    allocate(VAB(1:nn-2,1:2), VAB_T(1:2,1:nn-2), VA(1:nn-2,1:nn-2),NewCovMat(1:nn-2,1:nn-2))

    VA = mat(1:nn-2,1:nn-2)

    VAB = mat(1:nn-2,nn-1:nn)
    VAB_T = mat(nn-1:nn,1:nn-2)


    VB = mat(nn-1:nn,nn-1:nn) + identity(IdSize)
    detVB = VB(1,1)*VB(2,2) - VB(1,2)*VB(2,1)
    VB_inv = reshape((/VB(2,2), -VB(2,1), -VB(1,2), VB(1,1)/), (/2,2/))/detVB

    NewCovMat(:,:) = VA - matmul(VAB, matmul(VB_inv,VAB_T))

    deallocate(VAB, VAB_T, VA)
  end function NewCovMat

!  type(GaussianState) function NewState(state)
!    use kinds
!    use vars
! 
!    type(GaussianState), intent(in) :: state
!
!    NewState%V = NewCovMat(state%V)
!    NewState%rbar = NewMean(state%rbar, state%V)
!
!  end function NewState
!
!
!  pure function NewStateA(state)
!    use kinds
!    use vars
! 
!    type(GaussianState), intent(in) :: state
!    type(GaussianState) :: NewStateA
!
!    NewStateA%V = VA(state%V)
!    NewStateA%rbar = rA(state%rbar, state%V)
!
!  end function NewStateA

  subroutine Compute_q(mat, rbar, q)
    use kinds
    use vars

    integer(ip) :: i, nn, IdSize
    real(wp), dimension(:,:), intent(in) :: mat
    real(wp), dimension(2,2) :: VB, VB_inv
    
    real(wp), dimension(:), intent(in) :: rbar
    real(wp), dimension(:), allocatable :: rbar_B
    real(wp), dimension(1,1:2) :: rbar_B_T
    real(wp), intent(out) :: q
    real(wp) :: detVB, detVB_inv
    real(wp), dimension(1) :: tmp

    nn = sqrt(real(size(mat),wp))

    IdSize = 2

    allocate(rbar_B(1:2))

    rbar_B(1:2) = rbar(nn-1:nn)
    rbar_B_T(1,1:2) = rbar(nn-1:nn)

    VB = mat(nn-1:nn,nn-1:nn) + identity(IdSize)
    detVB = VB(1,1)*VB(2,2) - VB(1,2)*VB(2,1)

    VB_inv = reshape((/VB(2,2), -VB(2,1), -VB(1,2), VB(1,1)/), (/2,2/))/detVB
    detVB_inv = VB_inv(1,1)*VB_inv(2,2) - VB_inv(1,2)*VB_inv(2,1)

    tmp = -matmul(rbar_B_T,matmul(VB_inv, rbar_B))


    q = 2.0_wp*exp(tmp(1))/sqrt(detVB)

  end subroutine Compute_q


  pure function identity(nn)
    use kinds
    integer(ip), intent(in) :: nn

    integer(ip) :: i, j

    real(wp), dimension(1:nn, 1:nn) :: identity

    forall(i=1:nn, j=1:nn) identity(i,j) = (i/j)*(j/i)

  end function identity

end module TorontonianSamples
