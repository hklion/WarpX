/* Copyright 2019-2022 Yinjian Zhao & Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEHISTOGRAM2D_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEHISTOGRAM2D_H_

#include "ReducedDiags.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>
#include <vector>

/**
 * Reduced diagnostics that computes a 2D histogram over particles
 * for a quantity specified by the user in the input file using the parser.
 */
class ParticleHistogram2D : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ParticleHistogram2D(std::string rd_name);

    /// number of axes
    static constexpr int m_naxes = 2;

    /// normalization type
    int m_norm;

    /// number of bins
    std::vector<int> m_bin_num;

    /// selected species index
    int m_selected_species_id = -1;

    /// max and min bin values
    std::vector<amrex::Real> m_bin_max;
    std::vector<amrex::Real> m_bin_min;

    /// bin size
    std::vector<amrex::Real> m_bin_size;

    /// Parsers to read expressions for particle quantity from the input file.
    /// 7 elements are t, x, y, z, ux, uy, uz
    static constexpr int m_nvars = 7;
    std::vector<std::unique_ptr<amrex::Parser>> m_parser;

    /// Optional parser to filter particles before doing the histogram
    std::unique_ptr<amrex::Parser> m_parser_filter;

    /// Whether the filter is activated
    bool m_do_parser_filter = false;

    /**
     * This function computes a histogram of user defined quantity.
     *
     * @param[in] step current time step
     */
    virtual void ComputeDiags(int step) override final;

};

#endif
