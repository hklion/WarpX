/* Copyright 2021 Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef GET_VELOCITY_H_
#define GET_VELOCITY_H_

#include "VelocityProperties.H"

/** Get velocity at a point on the grid
 *
 * Class to get velocity at a point on the grid, either constant (m_velocity)
 * or a spatially varying value computed using the parser function (m_velocity_parser).
 * It also stores the direction of the velocity field. It provides the velocity information
 * held by the VelocityProperties instance passed to the constructor.
 */
struct GetVelocity
{
    /* Type of velocity initialization */
    VelocityInitType m_type;

    /* Velocity direction */
    int m_dir; //! Index x=0, y=1, z=2
    int m_sign_dir; //! Sign of the velocity direction positive=1, negative=-1

    /** Constant velocity value, if m_type == VelConstantValue */
    amrex::Real m_velocity;
    /** Velocity parser function, if m_type == VelParserFunction */
    amrex::ParserExecutor<3> m_velocity_parser;

    /**
     * \brief Construct the functor with information provided by vel
     *
     * \param[in] vel: const reference to the VelocityProperties object that will be used to
     * populate the functor
     */
    GetVelocity (VelocityProperties const& vel) noexcept;

    /**
     * \brief Functor call. Returns the value of velocity at the location (x,y,z)
     *
     * \param[in] x x-coordinate of given location
     * \param[in] y y-coordinate of given location
     * \param[in] z z-cooridnate of given location
     * \param[in] comp component of velocity to return; 0 for x, 1 for y, 2 for z
     *
     *\return value of velocity at (x,y,z).
     *        m_velocity if m_type is VelConstantValue
     *        m_velocity_parser(x,y,z) if m_type is VelParserFunction
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator() (amrex::Real const x, amrex::Real const y, amrex::Real const z, int comp) const noexcept
    {
        switch (m_type)
        {
            case (VelConstantValue):
            {
                return (m_dir == comp) ? m_sign_dir * m_velocity : 0.0;
            }
            case (VelParserFunction):
            {
                return (m_dir == comp) ? m_sign_dir * m_velocity_parser(x,y,z) : 0.0;
            }
            case (VelComponentsParserFunction):
            {
                if (comp == 0)
                {
                    return m_velocity_x_parser(x,y,z);
                }
                else if (comp == 1)
                {
                    return m_velocity_y_parser(x,y,z);
                }
                else if (comp == 2)
                {
                    return m_velocity_z_parser(x,y,z);
                }
                else
                {
                    WARPX_ABORT_WITH_MESSAGE("Unknown velocity component " + str(comp) + ".");
                    return 0.0;
                }
            }
            default:
            {
                amrex::Abort("Get initial velocity: unknown type");
                return 0.0;
            }
        }
    }

    /**
     * \brief Returns the index of the direction of the bulk velocity
     *
     *\return index of direction of velocity.
     *        0: x
     *        1: y
     *        2: z
     */
    AMREX_GPU_HOST_DEVICE
    int direction () const noexcept
    {
        return m_dir;
    }
};
#endif
