#include "heatdis.hpp"

namespace heatdis
{
  void initData(int nbLines, int M, int rank, Kokkos::View<double*> h) {

    typedef Kokkos::RangePolicy<> range_policy;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;

    /* set all of the data to 0 */
    Kokkos::parallel_for("init_h", mdrange_policy({0,0}, {nbLines, M}),
                         KOKKOS_LAMBDA (const int i, const int j) {
      h((i*M)+j) = 0;
    }
    );

    /* initialize some values on rank 0 to 100 */
    if (rank == 0) {
      int j = ceil(M * 0.9);
      Kokkos::parallel_for("init_rank0", range_policy((int)(M*0.1), j),
                           KOKKOS_LAMBDA (const int i) {
        h(i) = 100;
      }
      );
    }
  }

  double doWork(int numprocs, int rank, int M, int nbLines,
                      Kokkos::View<double*> g, Kokkos::View<double*> h) {

    MPI_Request req1[2], req2[2];
    MPI_Status status1[2], status2[2];
    double localerror;
    localerror = 0;

    using range_policy = Kokkos::RangePolicy<>;
    using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    Kokkos::parallel_for("copy_g", mdrange_policy({0,0}, {nbLines, M}),
                         KOKKOS_LAMBDA (const int i, const int j) {
      h((i*M)+j) = g((i*M)+j);
    }
    );

    double *g_raw = g.data();
    double *h_raw = h.data();

    /* Send and receive data from the left -- all ranks besides 0 because there are no ranks to its left */
    if (rank > 0) {
      MPI_Isend(g_raw+M, M, MPI_DOUBLE, rank-1, WORKTAG, MPI_COMM_WORLD, &req1[0]);
      MPI_Irecv(h_raw,   M, MPI_DOUBLE, rank-1, WORKTAG, MPI_COMM_WORLD, &req1[1]);
    }

    /* Send and receive data from the right -- all ranks besides numprocs - 1
     * because there are no ranks to its right */
    if (rank < numprocs - 1) {
      MPI_Isend(g_raw+((nbLines-2)*M), M, MPI_DOUBLE, rank+1, WORKTAG, MPI_COMM_WORLD, &req2[0]);
      MPI_Irecv(h_raw+((nbLines-1)*M), M, MPI_DOUBLE, rank+1, WORKTAG, MPI_COMM_WORLD, &req2[1]);
    }

    if (rank > 0) {
      MPI_Waitall(2,req1,status1);
    }
    if (rank < numprocs - 1) {
      MPI_Waitall(2,req2,status2);
    }

    /* perform the computation */
    Kokkos::parallel_for("compute", mdrange_policy({1,0}, {nbLines-1, M}),
                         [&localerror, &g, h, M] (const int i, const int j) {
                           g((i*M)+j) = 0.25*(h(((i-1)*M)+j)+h(((i+1)*M)+j)+h((i*M)+j-1)+h((i*M)+j+1));
                           if(localerror < fabs(g((i*M)+j) - h((i*M)+j))) {
                             localerror = fabs(g((i*M)+j) - h((i*M)+j));
                           }
                         }
    );

    /* perform computation on right-most rank */
    if (rank == (numprocs-1)) {
      Kokkos::parallel_for("compute_right", range_policy(0,M),
                           [&g, nbLines, M] (const int j) {
                             g(((nbLines-1)*M)+j) = g(((nbLines-2)*M)+j);
                           }
      );

    }

    return localerror;
  }
}