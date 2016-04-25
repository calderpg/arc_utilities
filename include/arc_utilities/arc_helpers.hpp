#include <stdlib.h>
#include <functional>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>
#include <type_traits>
#include <random>
#include <array>

#ifdef ENABLE_PARALLEL
#include <omp.h>
#endif

#ifndef ARC_HELPERS_HPP
#define ARC_HELPERS_HPP

// Branch prediction hints
// Figure out which compiler we have
#if defined(__clang__)
    /* Clang/LLVM */
    #define likely(x) __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
#elif defined(__ICC) || defined(__INTEL_COMPILER)
    /* Intel ICC/ICPC */
    #define likely(x) __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
#elif defined(__GNUC__) || defined(__GNUG__)
    /* GNU GCC/G++ */
    #define likely(x) __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
    /* Microsoft Visual Studio */
    /* MSVC doesn't support branch prediction hints. Use PGO instead. */
    #define likely(x) (x)
    #define unlikely(x) (x)
#endif

// Macro to disable unused parameter compiler warnings
#define UNUSED(x) (void)(x)

namespace arc_helpers
{
    template <typename T>
    inline T SetBit(const T current, const u_int32_t bit_position, const bool bit_value)
    {
        // Safety check on the type we've been called with
        static_assert((std::is_same<T, u_int8_t>::value
                       || std::is_same<T, u_int16_t>::value
                       || std::is_same<T, u_int32_t>::value
                       || std::is_same<T, u_int64_t>::value),
                      "Type must be a fixed-size unsigned integral type");
        // Do it
        T update_mask = 1;
        update_mask = update_mask << bit_position;
        if (bit_value)
        {
            return (current | update_mask);
        }
        else
        {
            update_mask = (~update_mask);
            return (current & update_mask);
        }
    }

    template<typename Datatype, typename Allocator=std::allocator<Datatype>>
    static Eigen::MatrixXd BuildDistanceMatrix(const std::vector<Datatype, Allocator>& data, std::function<double(const Datatype&, const Datatype&)>& distance_fn)
    {
        Eigen::MatrixXd distance_matrix(data.size(), data.size());
#ifdef ENABLE_PARALLEL
        #pragma omp parallel for schedule(guided)
#endif
        for (size_t idx = 0; idx < data.size(); idx++)
        {
            for (size_t jdx = 0; jdx < data.size(); jdx++)
            {
                distance_matrix(idx, jdx) = distance_fn(data[idx], data[jdx]);
            }
        }
        return distance_matrix;
    }

    class MultivariteGaussianDistribution
    {
    protected:
        const Eigen::VectorXd mean_;
        const Eigen::MatrixXd chol_decomp_;

        std::normal_distribution<double> unit_gaussian_dist_;

        template<typename Generator>
        inline Eigen::VectorXd Sample(Generator& prng)
        {
            Eigen::VectorXd draw;
            draw.resize(mean_.rows());

            for (ssize_t idx = 0; idx < draw.rows(); idx++)
            {
                draw(idx) = unit_gaussian_dist_(prng);
            }

            return chol_decomp_ * draw + mean_;
        }

    public:
        inline MultivariteGaussianDistribution(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) : mean_(mean), chol_decomp_(covariance.llt().matrixL().transpose()), unit_gaussian_dist_(0.0, 1.0)
        {
            assert(mean.rows() == covariance.rows());
            assert(covariance.cols() == covariance.rows());
        }

        template<typename Generator>
        inline Eigen::VectorXd operator()(Generator& prng)
        {
            return Sample(prng);
        }
    };

    class RandomRotationGenerator
    {
    protected:

        std::uniform_real_distribution<double> uniform_unit_dist_;

        // From: "Uniform Random Rotations", Ken Shoemake, Graphics Gems III, pg. 124-132
        template<typename Generator>
        inline Eigen::Quaterniond GenerateUniformRandomQuaternion(Generator& prng)
        {
            const double x0 = uniform_unit_dist_(prng);
            const double r1 = sqrt(1.0 - x0);
            const double r2 = sqrt(x0);
            const double t1 = 2.0 * M_PI * uniform_unit_dist_(prng);
            const double t2 = 2.0 * M_PI * uniform_unit_dist_(prng);
            const double c1 = cos(t1);
            const double s1 = sin(t1);
            const double c2 = cos(t2);
            const double s2 = sin(t2);
            const double x = s1 * r1;
            const double y = c1 * r1;
            const double z = s2 * r2;
            const double w = c2 * r2;
            return Eigen::Quaterniond(w, x, y, z);
        }

        // From Effective Sampling and Distance Metrics for 3D Rigid Body Path Planning, by James Kuffner, ICRA 2004
        template<typename Generator>
        Eigen::Vector3d GenerateUniformRandomEulerAngles(Generator& prng)
        {
            const double roll = M_PI * (-2.0 * uniform_unit_dist_(prng) + 1.0);
            const double pitch = acos(1.0 - 2.0 * uniform_unit_dist_(prng)) - M_PI_2;
            const double yaw = M_PI * (-2.0 * uniform_unit_dist_(prng) + 1.0);
            return Eigen::Vector3d(roll, pitch, yaw);
        }

    public:

        inline RandomRotationGenerator() : uniform_unit_dist_(0.0, 1.0) {}

        template<typename Generator>
        inline Eigen::Quaterniond GetQuaternion(Generator& prng)
        {
            return GenerateUniformRandomQuaternion(prng);
        }

        template<typename Generator>
        inline std::vector<double> GetRawQuaternion(Generator& prng)
        {
            const Eigen::Quaterniond quat = GenerateUniformRandomQuaternion(prng);
            return std::vector<double>{quat.x(), quat.y(), quat.z(), quat.w()};
        }

        template<typename Generator>
        inline Eigen::Vector3d GetEulerAngles(Generator& prng)
        {
            return GenerateUniformRandomEulerAngles(prng);
        }

        template<typename Generator>
        inline std::vector<double> GetRawEulerAngles(Generator& prng)
        {
            const Eigen::Vector3d angles = GenerateUniformRandomEulerAngles(prng);
            return std::vector<double>{angles.x(), angles.y(), angles.z()};
        }
    };
}

#endif // ARC_HELPERS_HPP
