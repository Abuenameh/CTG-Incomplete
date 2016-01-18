/* 
 * File:   casadi.hpp
 * Author: Abuenameh
 *
 * Created on 06 November 2014, 17:45
 */

#ifndef CASADI_HPP
#define	CASADI_HPP

#include <casadi/casadi.hpp>

using namespace casadi;

#include <boost/date_time.hpp>

using namespace boost::posix_time;

#include "gutzwiller.hpp"

SX energy(SX& fin, SX& J, SX& U0, SX& dU, SX& mu, SX& theta);
SX energy(int i, int n, SX& fin, SX& J, SX& U0, SX& dU, SX& mu, SX& theta);
    
#endif	/* CASADI_HPP */

