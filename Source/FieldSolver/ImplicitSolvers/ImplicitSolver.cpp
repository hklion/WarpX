#include "ImplicitSolver.H"
#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"

using namespace amrex;

void ImplicitSolver::CreateParticleAttributes () const
{
    // Set comm to false to that the attributes are not communicated
    // nor written to the checkpoint files
    int const comm = 0;

    // Add space to save the positions and velocities at the start of the time steps
    for (auto const& pc : m_WarpX->GetPartContainer()) {
#if (AMREX_SPACEDIM >= 2)
        pc->AddRealComp("x_n", comm);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        pc->AddRealComp("y_n", comm);
#endif
        pc->AddRealComp("z_n", comm);
        pc->AddRealComp("ux_n", comm);
        pc->AddRealComp("uy_n", comm);
        pc->AddRealComp("uz_n", comm);
    }
}

const Geometry& ImplicitSolver::GetGeometry (const int a_lvl) const
{
    AMREX_ASSERT((a_lvl >= 0) && (a_lvl < m_num_amr_levels));
    return m_WarpX->Geom(a_lvl);
}

const Array<FieldBoundaryType,AMREX_SPACEDIM>& ImplicitSolver::GetFieldBoundaryLo () const
{
    return m_WarpX->GetFieldBoundaryLo();
}

const Array<FieldBoundaryType,AMREX_SPACEDIM>& ImplicitSolver::GetFieldBoundaryHi () const
{
    return m_WarpX->GetFieldBoundaryHi();
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::GetLinOpBCLo () const
{
    return convertFieldBCToLinOpBC(m_WarpX->GetFieldBoundaryLo());
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::GetLinOpBCHi () const
{
    return convertFieldBCToLinOpBC(m_WarpX->GetFieldBoundaryHi());
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::convertFieldBCToLinOpBC (const Array<FieldBoundaryType,AMREX_SPACEDIM>& a_fbc) const
{
    Array<LinOpBCType, AMREX_SPACEDIM> lbc;
    for (auto& bc : lbc) { bc = LinOpBCType::interior; }
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
        if (a_fbc[i] == FieldBoundaryType::PML) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Periodic) {
            lbc[i] = LinOpBCType::Periodic;
        } else if (a_fbc[i] == FieldBoundaryType::PEC) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Damped) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Absorbing_SilverMueller) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Neumann) {
            // Also for FieldBoundaryType::PMC
            lbc[i] = LinOpBCType::symmetry;
        } else if (a_fbc[i] == FieldBoundaryType::PECInsulator) {
            ablastr::warn_manager::WMRecordWarning("Implicit solver",
                "With PECInsulator, in the Curl-Curl preconditioner Neumann boundary will be used since the full boundary is not yet implemented.",
                ablastr::warn_manager::WarnPriority::medium);
            lbc[i] = LinOpBCType::symmetry;
        } else if (a_fbc[i] == FieldBoundaryType::None) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Open) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else {
            WARPX_ABORT_WITH_MESSAGE("Invalid value for FieldBoundaryType");
        }
    }
    return lbc;
}
