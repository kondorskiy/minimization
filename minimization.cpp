
/*======================================================================================================================

  CLASS FOR FUNCTION MINIMIZATION USING CONJUGATED GRADIENT SEARCH.

  Notes:
    Contains subroutines adopted from [W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
      "Numerical Recipes in C The Art of Scientific Computing" (Cambridge University Press)].

  ACKNOWLEDGEMENT(S): Alexey D. Kondorskiy,
    P.N.Lebedev Physical Institute of the Russian Academy of Science.
    E-mail: kondorskiy@lebedev.ru, kondorskiy@gmail.com.

  Last modified: April 17, 2023.

======================================================================================================================*/

class Minimization
{

  /*--------------------------------------------------------------------------------------------------------------------
    Set up calculation parameters.
  --------------------------------------------------------------------------------------------------------------------*/

  // Minimal step to calculate the gradient of the function.
  private: double grad_step_min;

  // Relative first step for bracketing.
  private: double brack_step_ratio;

  // Maximal number of minimization iterations allowed.
  private: int max_iter_num;

  // Maximal number of bracketing steps allowed.
  private: int max_ls_brak_num;

  // Maximal number of accurate search steps allowed.
  private: int max_ls_fine_num;

  // Tolerance with respect to argument value.
  private: double arg_tol;

  // Tolerance with respect to function value.
  private: double fun_tol;

  // Choose the formula to calculate conjugate gradient true: Polak-Ribiere; false: Fletcher-Reeves.
  private: bool is_polak_ribiere;

  // Flag to restart conjugate gradient.
  private: bool restart_gradient;

  // Cache for minimal parameters: Minimal value of functional found.
  private: double c_min_val;

  // Cache for minimal parameters: Corresponding functional arguments.
  private: std::vector<double> c_min_arg;

  // Log file name.
  private: std::string log_name;

  // Flag to allow printing progress to log file.
  private: bool is_report;


  /*--------------------------------------------------------------------------------------------------------------------
    Constructor and destructor.
  --------------------------------------------------------------------------------------------------------------------*/
  public: Minimization(
    std::string log_name_i,                       // Log file name.
    bool is_report_i,                             // Flag to allow printing to log file.
    double scale)                                 // Step scale parameter.
  {
    log_name = log_name_i;
    is_report = is_report_i;
    grad_step_min = scale;
    brack_step_ratio = scale;
    max_iter_num = 1000;
    max_ls_brak_num = 1000;
    max_ls_fine_num = 1000;
    arg_tol = scale;
    fun_tol = scale;
    is_polak_ribiere = true;
  }

  public: ~Minimization(){};


  /*--------------------------------------------------------------------------------------------------------------------
    Clear cache.
  --------------------------------------------------------------------------------------------------------------------*/
  private: void gs_flushCache()
  {
    c_min_val = 1.0e100;
    c_min_arg.clear();
  }


  /*--------------------------------------------------------------------------------------------------------------------
    Clear cache.
  --------------------------------------------------------------------------------------------------------------------*/
  private: void gs_updateCache(
    double val,                                   // Value of functional.
    std::vector<double> arg)                      // Functional arguments.
  {
    int arg_num = arg.size();
    if (c_min_arg.size() == 0) {
      c_min_val = val;
      for (int i = 0; i < arg_num; ++i)
        c_min_arg.push_back(arg[i]);
    } else
      if (c_min_val > val) {
        c_min_val = val;
        for (int i = 0; i < arg_num; ++i)
          c_min_arg[i] = arg[i];
      }
  }


  /*--------------------------------------------------------------------------------------------------------------------
    Gradient of the function to be minimized by finite difference.
  --------------------------------------------------------------------------------------------------------------------*/
  private: void gs_getGradient(
    double (*func)(std::vector<double> &),        // Function to minimize.
    std::vector<double> arg,                      // Functional arguments.
    std::vector<double> steps,                    // Differentiation steps.
    std::vector<bool> var_list,                   // Flags to allow variation.
    std::vector<double> &grad)                    // Result: functional derivatives.
  {
    int i, j, arg_num = arg.size();
    double centr, tmp, grad_step;
    std::vector<double> tmp_arg(arg_num);

    // Calculate function for arguments without shift.
    centr = (*func)(arg);
    gs_updateCache(centr, arg);

    // Cycle with respect to the shifts of the arguments.
    for (i = 0; i < arg_num; ++i) {
      if (var_list[i]) {

        // Set differentiation step.
        if (abs(arg[i]) > 0.0)
          grad_step = abs(arg[i])*steps[i];
        else
          grad_step = grad_step_min;

        // Get component of the gradient.
        for (j = 0; j < arg_num; j++)
          tmp_arg[j] = arg[j];
        tmp_arg[i] = arg[i] + grad_step;
        double ftmp = (*func)(tmp_arg);
        gs_updateCache(ftmp, tmp_arg);
        grad[i] = (ftmp - centr)/grad_step;
      } else
        grad[i] = 0.0;
    }
  }


  /*--------------------------------------------------------------------------------------------------------------------
    Calculation of the conjugate gradient.
  --------------------------------------------------------------------------------------------------------------------*/
  private: void gs_makeConjugateGradient(
    std::vector<double> gp,                       // Previous iteration of gradient.
    std::vector<double> gn,                       // Current iteration of gradient.
    std::vector<double> &h)                       // Result conjugate gradient.
  {
    int i, arg_num = gp.size();
    double beta, beta_nomer, beta_denom;

    beta_nomer = 0.0;
    beta_denom = 0.0;
    for (i = 0; i < arg_num; ++i) {
      if (is_polak_ribiere)
        beta_nomer = beta_nomer + gn[i]*(gn[i] - gp[i]);
      else
        beta_nomer = beta_nomer + gn[i]*gn[i];
      beta_denom = beta_denom + gp[i]*gp[i];
    }

    if (beta_denom == 0.0) 
      beta = 0.0;
    else
      beta = beta_nomer/beta_denom;

    for (i = 0; i < arg_num; ++i)
      h[i] = gn[i] + beta*h[i];
  }


  /*--------------------------------------------------------------------------------------------------------------------
    Some trivial mathematics to keep with this code.
  --------------------------------------------------------------------------------------------------------------------*/

  // Swap two values.
  private: inline void gs_swap(double &a, double &b)
    { double c = a; a = b; b = c; }

  // Get maximum of two values.
  private: inline double gs_max(double a, double b)
    { if(a > b) return a; else return b; }

  // Conditioned sign.
  private: inline double gs_sgn(double a, double b)
    { if(b >= 0.0) return abs(a); else return (-1.0)*abs(a); }

  // Convert int to string.
  private: std::string intToString(const int &i)
    { std::stringstream ss; ss << i; return ss.str(); }


  /*--------------------------------------------------------------------------------------------------------------------
    Find bracketing triplet for the function minimum along conjugate gradient direction.
  --------------------------------------------------------------------------------------------------------------------*/
  private: void gs_mnbrak(
    double (*func)(std::vector<double> &),        // Function to minimize.
    std::vector<double> f,                        // Array of function arguments.
    std::vector<double> h,                        // Conjugate gradient.
    double &ax, double &fa,                       // 1st point of bracketing triplet.
    double &bx, double &fb,                       // 2nd point of bracketing triplet.
    double &cx, double &fc,                       // 3rd point of bracketing triplet.
    int ls_brak_num)                              // Maximal number of steps allowed.
  {
    bool flag = true;
    int k, ils = 0, arg_num = f.size();
    const double GOLD = 1.618034, GLIMIT = 100.0, TINY = 1.0E-20;
    double ulim, u, r, q, fu, ftmp, ls_step, tmp;
    std::vector<double> arg(arg_num);

    tmp = 0.0;
    for (k = 0; k < arg_num; k++)
      tmp = tmp + h[k]*h[k];

    if (tmp == 0.0) 
      ls_step = grad_step_min;
    else {
      tmp = sqrt(tmp);
      ls_step = 0.0;
      for (k = 0; k < arg_num; k++)
        ls_step = ls_step + f[k]*h[k]*brack_step_ratio/tmp;
      ls_step = abs(ls_step);
      if (ls_step == 0.0) ls_step = grad_step_min;
    }

    ax = 0.0;
    fa = (*func)(f);
    gs_updateCache(fa, f);

    bx = ls_step;
    for (k = 0; k < arg_num; k++)
      arg[k] = f[k] + bx*h[k];
    fb = (*func)(arg);
    gs_updateCache(fb, arg);

    if (fb > fa) {
      gs_swap(ax, bx);
      gs_swap(fb, fa);
    }
    cx = bx + GOLD*(bx - ax);

    for (k = 0; k < arg_num; k++)
      arg[k] = f[k] + cx*h[k];
    fc = (*func)(arg);
    gs_updateCache(fc, arg);

    while ((fb > fc) && flag) {
      r = (bx - ax)*(fb - fc);
      q = (bx - cx)*(fb - fa);
      u = bx - ((bx - cx)*q - (bx - ax)*r)/(2.0*gs_sgn(gs_max(abs(q - r), TINY), q - r));
      ulim = bx + GLIMIT*(cx - bx);

      if ((bx - u)*(u - cx) > 0.0) {
        for (k = 0; k < arg_num; k++)
          arg[k] = f[k] + u*h[k];
        fu = (*func)(arg);
        gs_updateCache(fu, arg);
        if (fu < fc) {
          ax = bx; bx = u; fa = fb; fb = fu;
          return;
        } else
          if (fu > fb) {
            cx = u; fc = fu;
            return;
        }
        u = cx + GOLD*(cx - bx);
        for (k = 0; k < arg_num; k++)
          arg[k] = f[k] + u*h[k];
        fu = (*func)(arg);
        gs_updateCache(fu, arg);
      } else {
        if ((cx - u)*(u - ulim) > 0.0) {
          for (k = 0; k < arg_num; k++)
            arg[k] = f[k] + u*h[k];
          fu = (*func)(arg);
          gs_updateCache(fu, arg);
          if (fu < fc) {
            bx = cx; cx = u;
            u = cx + GOLD*(cx - bx);
            fb = fc; fc = fu;
            for (k = 0; k < arg_num; k++)
              arg[k] = f[k] + u*h[k];
            fu = (*func)(arg);
            gs_updateCache(fu, arg);
          }
        } else {
          if ((u - ulim)*(ulim - cx) >= 0.0) {
            u = ulim;
            for (k = 0; k < arg_num; k++)
              arg[k] = f[k] + u*h[k];
            fu = (*func)(arg);
            gs_updateCache(fu, arg);
          } else {
            u = cx + GOLD*(cx - bx);
            for (k = 0; k < arg_num; k++)
              arg[k] = f[k] + u*h[k];
            fu = (*func)(arg);
            gs_updateCache(fu, arg);
          }
        }
      }
      ax = bx; bx = cx; cx = u;
      fa = fb; fb = fc; fc = fu;

      ils++;
      if (ils == ls_brak_num)
        flag = false;
    }
  }


  /*--------------------------------------------------------------------------------------------------------------------
    Get accurate function minimum along conjugate gradient direction.
  --------------------------------------------------------------------------------------------------------------------*/
  private: double gs_findAccurate(
    double (*func)(std::vector<double> &),        // Function to minimize,
    std::vector<double> f,                        // Array of function arguments.
    std::vector<double> h,                        // Conjugate gradient.
    int ls_fine_num,                              // Maximal number of steps allowed.
    double &a, double &fa,                        // 1st point of bracketing triplet.
    double &b, double &fb,                        // 2nd point of bracketing triplet.
    double &c, double &fc)                        // 3rd point of bracketing triplet.
  {
    bool flag1, flag = true;
    int k, i = 0, arg_num = f.size();
    double u, fu, v, fv;
    std::vector<double> arg(arg_num);

    // Cycle for iterations.
    while (flag) {
      u = 0.5*(a + b);
      for (k = 0; k < arg_num; k++)
        arg[k] = f[k] + u*h[k];
      fu = (*func)(arg);
      gs_updateCache(fu, arg);

      v = 0.5*(b + c);
      for (k = 0; k < arg_num; k++)
        arg[k] = f[k] + v*h[k];
      fv = (*func)(arg);
      gs_updateCache(fv, arg);

      // Get smaller triplet.
      flag1 = true;
      if (flag1 && (fu <= fa) && (fu <= fb)) {
        c = b; fc = fb; b = u; fb = fu; flag1 = false;
      }
      if (flag1 && (fb <= fu) && (fb <= fv)) {
        a = u; fa = fu; c = v; fc = fv; flag1 = false;
      }
      if (flag1 && (fv <= fb) && (fv <= fc)) {
        a = b; fa = fb; b = v; fb = fv; flag1 = false;
      }

      // Go out if 
      // if (abs((fa - fc)/fb) < fun_tol) flag = false;
      if (abs((a - c)/b) < arg_tol)
        flag = false;
      i++;

      if (i == ls_fine_num)
        flag = false;
    }

    return b;
  }


  /*--------------------------------------------------------------------------------------------------------------------
    Get minimum of function.
  --------------------------------------------------------------------------------------------------------------------*/
  public: void findMinimum(
    double (*func)(std::vector<double> &),        // Function to minimize.
    std::vector<double> &arg,                     // Function arguments to optimize.
    std::vector<bool> var_list,                   // Flags to allow variation.
    std::vector<double> steps)                    // Differentiation steps.
  {
    bool flag = true;
    int i, iter = 0, arg_num = arg.size();
    double a, a0, a1, a2, J0, J1, J2, fun0, fun1, tmp;
    std::vector<double> f(arg_num), gn(arg_num), gp(arg_num), h(arg_num);

    // Argument size check.
    if ( (arg.size() != var_list.size()) || (var_list.size() != steps.size()) ) {
      std::cout << "In fitData argument array sizes missmatch!\n";
      exit(0);
    }

    // Initial settings.
    for (i = 0; i < arg_num; ++i) {
      f[i] = arg[i];
      gp[i] = 0.0;
      h[i] = 0.0;
    }

    // Cycle on iterations.
    fun0 = (*func)(arg);

    // Ordinary start.
    restart_gradient = false;
    while (flag) {
      iter++;

      // Flush cache.
      gs_flushCache();

      // Calculate gradient.
      gs_getGradient(func, f, steps, var_list, gn);

      // Calculate conjugate gradient.
      if ( (iter == 0) || restart_gradient )
        for (i = 0; i < arg_num; ++i)
          h[i] = gn[i];
      else
        gs_makeConjugateGradient(gp, gn, h);

      // Perform linear search.
      gs_mnbrak(func, f, h, a0, J0, a1, J1, a2, J2, max_ls_brak_num);
      a = gs_findAccurate(func, f, h, max_ls_fine_num, a0, J0, a1, J1, a2, J2);

      // Prepare next iteration.
      for (i = 0; i < arg_num; ++i) {
        f[i] = f[i] + a*h[i]; 
        arg[i] = f[i]; 
        gp[i] = gn[i];
      }

      // Check iteration results.
      double fun1;
      fun1 = (*func)(arg);
      gs_updateCache(fun1, arg);
      if (abs(fun1) > 0.0) {

        // Check exit conditions. Exit by function tolerance.
        if (abs((fun1 - fun0)/fun1) < fun_tol)
          flag = false;

        // Check if there is smaller value in the cache.
        if (abs((c_min_val - fun1)/fun1) > fun_tol) {
          for (i = 0; i < arg_num; ++i)
            arg[i] = c_min_arg[i];
          fun0 = c_min_val;
          restart_gradient = true;
        } else
          restart_gradient = false;
      }
      fun0 = fun1;

      // Exceptional exit.
      if (iter == max_iter_num) flag = false;

      // Report.
      if (is_report) {
        std::ofstream fout(log_name.c_str(), std::ios::app);
        fout << "\n AFTER ITERATION : " << iter << "  Residual value = " << fun0 << "\n";
        fout.close();
      }
    }
  }

};

//======================================================================================================================
