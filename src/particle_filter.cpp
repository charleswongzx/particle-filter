/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 10;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2]; // Standard deviations for x, y, and theta

    // random distribution about mean
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    // assign to each particle
    for (int i = 0; i < num_particles; i++) {
        Particle new_particle;
        new_particle.x = dist_x(gen);
        new_particle.y = dist_y(gen);
        new_particle.theta = dist_theta(gen);
        new_particle.weight = 1;

        // push particle onto particles list
        particles.push_back(new_particle);
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // define normal distributions for sensor noise
    normal_distribution<double> N_x(0, std_pos[0]);
    normal_distribution<double> N_y(0, std_pos[1]);
    normal_distribution<double> N_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {

        // calculate new state
        if (fabs(yaw_rate) < 0.00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }

        // add noise
        particles[i].x += N_x(gen);
        particles[i].y += N_y(gen);
        particles[i].theta += N_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (auto& obs: observations) {

        // grab current observation
        LandmarkObs o = obs;

        // init minimum distance to maximum possible
        double min_dist = numeric_limits<double>::max();

        // init id of landmark from map placeholder to be associated with the observation
        int map_id = -1;

        for (auto& prediction: predicted) {
            // grab current prediction
            LandmarkObs p = prediction;

            // get distance between current/predicted landmarks
            double cur_dist = dist(o.x, o.y, p.x, p.y);

            // find the predicted landmark nearest the current observed landmark
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                map_id = p.id;
            }
        }

        // set the observation's id to the nearest predicted landmark's id
        obs.id = map_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for(auto& p: particles){
        p.weight = 1.0;

        // filter detect-able landmarks
        vector<LandmarkObs> predictions;
        for(const auto& lm: map_landmarks.landmark_list){
            double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
            if( distance < sensor_range){ // if the landmark is within the sensor range, save it to predictions
                predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
            }
        }

        // car > map coordinate conversion
        vector<LandmarkObs> observations_map;
        double cos_theta = cos(p.theta);
        double sin_theta = sin(p.theta);

        for(const auto& obs: observations){
            LandmarkObs tmp;
            tmp.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
            tmp.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
            //tmp.id = obs.id; // maybe an unnecessary step, since the each obersation will get the id from dataAssociation step.
            observations_map.push_back(tmp);
        }

        dataAssociation(predictions, observations_map);

        //set weight for each particle
        for(const auto& obs_m: observations_map){

            Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
            double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
            double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
            double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
            p.weight *=  w;
        }

        weights.push_back(p.weight);

    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // generate distribution according to weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(weights.begin(), weights.end());

    // create resampled particles
    vector<Particle> resampled_particles;
    resampled_particles.resize(num_particles);

    // resample the particles according to weights
    for(int i=0; i<num_particles; i++){
        int idx = dist(gen);
        resampled_particles[i] = particles[idx];
    }

    // assign the resampled_particles to the previous particles
    particles = resampled_particles;

    // clear the weight vector for the next iteration
    weights.clear();

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
