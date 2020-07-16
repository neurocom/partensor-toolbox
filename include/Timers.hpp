#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      25/04/2019
* @author    Technical University of Crete team:
*            Athanasios P. Liavas
*            Paris Karakasis
*            Christos Kolomvakis
*            John Papagiannakos
*            Siaminou Nina
* @author    Neurocom SA team:
*            Christos Tsalidis
*            Georgios Lourakis
*            George Lykoudis
*/
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      Timers.hpp
* @details
* Implements the following time functions, 
*
*  - @c clock,
*  - @c std::chrono with both steady and high resolution clock,
*  - @c MPI_Wtime.
********************************************************************/

#ifndef PARTENSOR_TIMERS_HPP
#define PARTENSOR_TIMERS_HPP

#include <chrono>
#include "mpi.h"

namespace partensor {
  
  /**
   * Struct with implementations for measuring time from 
   * libraries as @c time, @c stl chrono, and @c MPI.
   */
  struct Timers
  {
    using ClockSteady = std::chrono::time_point<std::chrono::steady_clock>;          /**< Typdef for chrono steady clock. */
    using ClockHigh   = std::chrono::time_point<std::chrono::high_resolution_clock>; /**< Typdef for chrono high resolution clock. */

    clock_t     cpu_current_time;
    double 	    cpu_elapsed_time;

    ClockHigh   chrono_high_current_time;
    double      chrono_high_elapsed_time;

    ClockSteady chrono_steady_current_time;
    double      chrono_steady_elapsed_time;

    double      mpi_current_time;
    double 	    mpi_elapsed_time;

    /**
     * Stores in a variable of @c clock_t type, the processor's time.
     */
    void startCpuTimer ()
    {
      cpu_current_time = clock();
    }

    /**
     * Stores in a variable of @c double type, the seconds passed since 
     * @c startCpuTimer was called.
     * 
     * @returns The measured time passed.
     */
    double endCpuTimer ()
    {
      auto finish      = clock();
      cpu_elapsed_time = (float)(finish - cpu_current_time)/(float)CLOCKS_PER_SEC ;
      return cpu_elapsed_time;
    }

    /**
     * Stores in a variable of @c chrono @c high_resolution_clock type, the current time.
     */
    void startChronoHighTimer ()
    {
      chrono_high_current_time = std::chrono::high_resolution_clock::now();
    }

    /**
     * Stores in a variable of @c double type, the seconds passed since 
     * @c startChronoHighTimer was called.
     * 
     * @returns The measured time passed.
     */
    double endChronoHighTimer ()
    {
      std::chrono::duration<double> finish = std::chrono::high_resolution_clock::now() - chrono_high_current_time;
      chrono_high_elapsed_time = finish.count();
      return chrono_high_elapsed_time;
    }

    /**
     * Stores in a variable of @c chrono @c steady_clock type, the current point in time.
     */
    void startChronoSteadyTimer ()
    {
      chrono_steady_current_time = std::chrono::steady_clock::now();
    }

    /**
     * Stores in a variable of @c double type, the seconds passed since 
     * @c startChronoSteadyTimer was called.
     * 
     * @returns The measured time passed.
     */
    double endChronoSteadyTimer ()
    {
      std::chrono::duration<double> finish = std::chrono::steady_clock::now() - chrono_steady_current_time;
      chrono_steady_elapsed_time = finish.count();
      return chrono_steady_elapsed_time;
    }

    /**
     * Stores in a variable of @c double type, the elapsed time on the calling @c MPI processor.
     */
    void startMpiTimer ()
    {
      mpi_current_time = MPI_Wtime();
    }

    /**
     * Stores in a variable of @c double type, the seconds passed since 
     * @c startMpiTimer was called.
     * 
     * @returns The measured time passed.
     */
    double endMpiTimer ()
    {
      auto finish = MPI_Wtime();
      mpi_elapsed_time = (finish - mpi_current_time);
      return mpi_elapsed_time;
    }

  };

  static inline Timers timer; /**< A class @c Timers object */

} // end namespace partensor


#endif // end of PARTENSOR_TIMERS_HPP
