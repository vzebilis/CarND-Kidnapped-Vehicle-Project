/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <cassert>
#include <limits>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  default_random_engine gen;

  // Create the normal distribitions to pull initial values for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 70;  // TODO: Set the number of particles
  particles.reserve(num_particles);
  for (int i = 0; i < num_particles; ++i) {
    Particle prt;
    prt.id = i;
    prt.x = dist_x(gen);
    prt.y = dist_y(gen);
    prt.theta = dist_theta(gen);
    prt.weight = 1;
    particles.emplace_back(std::move(prt));
    weights.push_back(1);
  } 
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  if (yaw_rate == 0.0) {
    double vdt = velocity * delta_t;
    for (auto & prt: particles) {
      prt.x += vdt * cos(prt.theta);
      prt.y += vdt * sin(prt.theta);
    }
  } else {
    double vyaw = velocity / yaw_rate;
    double yawdt = yaw_rate * delta_t;
    for (auto & prt: particles) {
      prt.x += vyaw * (sin(prt.theta + yawdt) - sin(prt.theta));
      prt.y += vyaw * (cos(prt.theta) - cos(prt.theta + yawdt));
      prt.theta += yawdt;
    }
  }

  default_random_engine gen;
  // Add noise to particles (for x, y and theta)
  for (auto & prt: particles) {
    // Create the normal distribitions to pull initial values for x, y and theta
    normal_distribution<double> dist_x(prt.x, std_pos[0]);
    normal_distribution<double> dist_y(prt.y, std_pos[1]);
    normal_distribution<double> dist_theta(prt.theta, std_pos[2]);
    prt.x = dist_x(gen);
    prt.y = dist_y(gen);
    prt.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(const vector<Map::single_landmark_s> & predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  assert(predicted.size() and observations.size());

  for (auto & lmrk: observations) {
    Map::single_landmark_s closest;
    double minDist = std::numeric_limits<double>::max();
    for (auto pit = predicted.begin(); pit != predicted.end(); ++pit) {
      auto & pred = *pit;
      double curDist = dist(lmrk.x, lmrk.y, pred.x_f, pred.y_f);
      if(curDist < minDist) {
        closest = pred;
        minDist = curDist;
      }
    }
    assert(minDist > 0);
    // Assign the predicted landmark to the closest real one
    lmrk.id = closest.id_i;
    lmrk.x = closest.x_f;
    lmrk.y = closest.y_f;
  }
}

// Multivariate Gaussian probability function
static double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent = (pow(x_obs - mu_x, 2) / (2.0 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2.0 * pow(sig_y, 2)));

  // calculate weight using normalization terms and exponent
  double weight = gauss_norm * exp(-exponent);
  assert(weight >= 0);
  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // Update the weights of each particle and also determine max new weight
  maxWeight = -1;
  auto wit = weights.begin();
  for (auto & prt: particles) {

    // Transform the observations which are in vehicle coordinates into map coords
    // based on the location/heading of this particle 
    vector<LandmarkObs> mapObs;
    for (auto & obs: observations) {
      LandmarkObs mobs;
      mobs.id = obs.id;
      mobs.x = obs.x * cos(prt.theta) - obs.y * sin(prt.theta) + prt.x;
      mobs.y = obs.x * sin(prt.theta) + obs.y * cos(prt.theta) + prt.y;
      mapObs.emplace_back(std::move(mobs));
    }

    // Get the associated landmarks closest to these observations
    vector<LandmarkObs> mapClosest = mapObs; // copy mapObs
    dataAssociation(map_landmarks.landmark_list, mapClosest);

    vector<int> association_ids;
    vector<double> association_x;
    vector<double> association_y;
    prt.weight = 1;
    // Now go over again the observations and for each one compute the probability 
    // of observing such a distance from the expected landmark 
    // The weight of this particle is the product of these probabilities
    for (size_t i = 0; i < mapObs.size(); ++i) {
      auto & obs  = mapObs[i];
      auto & lmrk = mapClosest[i];
      // Call the multivariate Gaussian probability function and update the weight
      prt.weight *= multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, lmrk.x, lmrk.y);
      // Set the associations versus the real closest landmark in world coords
      association_ids.push_back(lmrk.id);
      association_x.push_back(lmrk.x);
      association_y.push_back(lmrk.y);
    }
    maxWeight = std::max(maxWeight, prt.weight);
    *wit = prt.weight;
    ++wit;
    // Add the associations needed
    SetAssociations(prt, association_ids, association_x, association_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Sanity check for the internal data structures
  assert(particles.size() == weights.size());
  
  // Sanity check that there is at least one positive weight
  {
    bool ok = false;
    for (double weight: weights) {
      if (weight != 0) { ok = true; break; }
    }
    if (!ok) std::cout << "No non-zero weight left in the weight vector!" << std::endl;
    assert(ok);
  }

  size_t N = particles.size();
  vector<Particle> samples; 
  samples.reserve(N); // reserve the space we know we will need
  default_random_engine gen;

  // Create the uniform distributions to use for sampling
  std::uniform_int_distribution<> uniform_distN(0, N - 1);
  std::uniform_real_distribution<> uniform_dist2maxWeight(0, 2 * maxWeight);
  int idx = uniform_distN(gen);
  double beta = 0.0;
  // We will select N samples, with resampling, based on a sample wheel concept
  for (size_t i = 0; i < N; ++i) {
    beta += uniform_dist2maxWeight(gen);
    while (weights[idx] < beta) {
      beta -= weights[idx];
      ++idx;
      idx %= N;
    }
    samples.emplace_back(particles[idx]); // copy the sampled particle
  }
  // Overwrite the internal particles vector with the sampled one
  particles = std::move(samples);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
