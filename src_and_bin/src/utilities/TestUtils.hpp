#include<algorithm>
#include<iostream>
#include<stdlib.h>
#include<math.h>
#include<utility>
#include<cmath>
#include<limits>

namespace utils{
    double mean(double * values, int size){
        double sum{0.0};
        for (int i = 0; i < size; ++i) 
            sum += values[i];
        return (sum / size);
    }

    std::pair<int, int> get_indices(int size, float quantile){
        float pos = size * quantile;

        std::pair<int, int> indices;

        // sets minimum index to 0 and adjusts index position
        if (pos < 1.0f){
            pos = 0.0f;
        } else{
            pos -= 1.0f;
        }

        if (fabs(std::fmod(pos, 1.0)) < std::numeric_limits<float>::epsilon()){
            // if pos is integer => use only single position
            indices.first = (int) pos;
            indices.second = (int) pos;
        } else {
            // use the mean between the two positions
            indices.first = floor(pos);
            indices.second = ceil(pos);
        }

        return indices;
    }

    std::pair<double, double> errors(double * values, int size, float quantile){
        // sort values in-place
        std::sort(values, values+size);


        // determine indices of lower quantile
        std::pair<int, int> indices = get_indices(size, quantile);
        std::pair<double, double> errors;

        errors.first = (values[indices.first] + values[indices.second]) / 2.0;

        // now determine indices of upper quantile
        indices = get_indices(size, 1.0 - quantile);

        errors.second = (values[indices.first] + values[indices.second]) / 2.0;

        return errors;
    }

}
