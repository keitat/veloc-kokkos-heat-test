#include <cxxopts/cxxopts.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>

#include "heatdis.hpp"
#include <resilience/CheckpointFilter.hpp>

using namespace heatdis;

/*
    This sample application is based on the heat distribution code
    originally developed within the FTI project: github.com/leobago/fti
*/

int main(int argc, char *argv[]) {
  int rank, nbProcs, nbLines, M;
  double wtime, memSize, localerror, globalerror = 1;

  auto options = cxxopts::Options("heatdis", "Sample heat distribution code");
  options.add_options()
             ("s,size", "Problem size", cxxopts::value<std::size_t>()->default_value("100"))
             ("n,nsteps", "Number of timesteps", cxxopts::value<std::size_t>()->default_value("600"))
             ("p,precision", "Min precision", cxxopts::value<double>()->default_value("0.00001"))
             ("c,checkpoint-interval", "Checkpoint interval", cxxopts::value<int>()->default_value("100"))
             ("config", "Config file", cxxopts::value<std::string>())
             ;

  options.parse_positional({"config"});
  auto args = options.parse(argc, argv);

  std::size_t nsteps = args["nsteps"].as< std::size_t >();
  const auto precision = args["precision"].as< double >();
  const auto chk_interval = args["checkpoint-interval"].as< int >();

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t mem_size = args["size"].as< std::size_t >();

  if (mem_size == 0) {
    printf("Wrong memory size! See usage\n");
    exit(3);
  }

  Kokkos::initialize(argc, argv);
  {
  auto ctx = KokkosResilience::make_context( MPI_COMM_WORLD, args["config"].as< std::string >() );

  const auto filt = KokkosResilience::Filter::NthIterationFilter( chk_interval );

  M = (int)sqrt((double)(mem_size * 1024.0 * 1024.0 * nbProcs) / (2 * sizeof(double))); // two matrices needed
  nbLines = (M / nbProcs) + 3;

  Kokkos::View<double*> h_view("h", M * nbLines);
  Kokkos::View<double*> g_view("g", M * nbLines);

  initData(nbLines, M, rank, g_view);

  memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

  if (rank == 0)
    printf("Local data size is %d x %d = %f MB (%lu).\n", M, nbLines, memSize, mem_size);
  if (rank == 0)
    printf("Target precision : %f \n", precision);
  if (rank == 0)
    printf("Maximum number of iterations : %lu \n", nsteps);

  wtime = MPI_Wtime();
  int i = 1 + KokkosResilience::latest_version(*ctx, "test_kokkos");

  while(i < nsteps) {

    KokkosResilience::checkpoint(*ctx, "test_kokkos", i, [&localerror, i, &globalerror,
                                                         g_view, h_view, nbProcs, rank, M, nbLines]() {
      localerror = doWork(nbProcs, rank, M, nbLines, g_view, h_view);

      if (((i % ITER_OUT) == 0) && (rank == 0)) {
        printf("Step : %d, error = %f\n", i, globalerror);
      }
      if ((i % REDUCED) == 0) {
        MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      }
    }, filt );

    if (globalerror < precision) {
      printf("PRECISION ERROR\n");
      break;
    }
    i++;
  }
  if (rank == 0)
    printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);

  }
  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
