
/*====================================================================

  TEST PROGRAM FOR CLASS FOR FUNCTION MINIMIZATION USING
    CONJUGATED GRADIENT SEARCH (MINIMIZATION.CPP).

  ACKNOWLEDGEMENT(S): Alexey D. Kondorskiy,
    P.N.Lebedev Physical Institute of the Russian Academy of Science.
    E-mail: kondorskiy@lebedev.ru, kondorskiy@gmail.com.

  Last modified: May 11, 2021.

====================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <math.h>
#include <iostream>
#include <vector>

#include "minimization.cpp"


/*--------------------------------------------------------------------
  Function to be minimized
--------------------------------------------------------------------*/
double testFunction(std::vector<double> &args)
{
  double res = 0.0;
  for(int i = 0; i < args.size(); i++)
    res = res + args[i]*args[i]*args[i]*args[i];
  return res;
}


/*********************************************************************
  TEST OF MINIMIZATION LIBRARY.
*********************************************************************/
int main(void)
{
  std::vector<double> arg;
  std::vector<bool> var_flags;
  std::vector<double> steps;

  /* Set initial guesses, flags to allow argument variation and
     differentiation step. Since the sensitivity of a function
     to variation of arguments differs from argument to argument,
     differentiation steps should be adjusted individually
     for each argument.  */

  // 1st argument.
  arg.push_back( 1.1);
  var_flags.push_back(true);
  steps.push_back(1.0e-3);

  // 2nd argument.
  arg.push_back(-0.4);
  var_flags.push_back(true);
  steps.push_back(1.0e-3);

  // 3rd argument.
  arg.push_back(-2.0);
  var_flags.push_back(true);
  steps.push_back(1.0e-3);

  // 4th argument.
  arg.push_back(0.01);
  var_flags.push_back(true);
  steps.push_back(1.0e-3);

  // 5th argument.
  arg.push_back(-0.9);
  var_flags.push_back(true);
  steps.push_back(1.0e-3);

  // Run!
  std::cout << "Initial guess : \n";
  for (int i = 0; i < arg.size(); i++)
    std::cout << "    " << arg[i] << "\n";
  std::cout << "\n";

  // Log file name.
  const std::string log_file_name = "minimization_progress.txt";

  // Run minimization.
  double (*func)(std::vector<double> &);
  func = &testFunction;
  Minimization my_min(log_file_name, true, 1.0e-8);
  my_min.findMinimum(func, arg, var_flags, steps);

  std::cout << "Minimum found at : \n";
  for (int i = 0; i < arg.size(); i++)
    std::cout << "    " << arg[i] << "\n";
  std::cout << "\n";

  return 0;
}
