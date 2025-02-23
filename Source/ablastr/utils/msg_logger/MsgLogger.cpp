/* Copyright 2021 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "MsgLogger.H"

#include "ablastr/utils/TextMsg.H"
#include "ablastr/utils/Serialization.H"

#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>

namespace abl_msg_logger = ablastr::utils::msg_logger;
namespace abl_ser = ablastr::utils::serialization;
namespace abl_utils = ablastr::utils;

using namespace abl_msg_logger;

#ifdef AMREX_USE_MPI
// Helper functions used only in this source file
namespace
{
    /**
    * \brief This collective function returns the messages of the "gather rank"
    * as a byte array.
    *
    * @param[in] my_msgs the messages of the current rank
    * @param[in] gather_rank the ID of the "gather rank"
    * @param[in] my_rank the ID of the current rank
    * @return the messages of the "gather rank" as a byte array
    */
    std::vector<char>
    get_serialized_gather_rank_msgs(
        const std::vector<Msg>& my_msgs,
        const int gather_rank,
        const int my_rank);

    /**
    * \brief This function generates data to send back to the "gather rank"
    *
    * @param[in] serialized_gather_rank_msgs the serialized messages of the gather rank
    * @param[in] gather_rank_how_many_msgs number of messages of the "gather rank"
    * @param[in] my_msg_map messages and counters of the current rank (as a map)
    * @param[in] is_gather_rank true if the rank is the "gather rank", false otherwise
    * @return a byte array to send back to the "gather rank" (or a dummy vector in case is_gather_rank is true)
    */
    std::vector<char>
    compute_package_for_gather_rank(
        const std::vector<char>& serialized_gather_rank_msgs,
        const std::int64_t gather_rank_how_many_msgs,
        const std::map<Msg, std::int64_t>& my_msg_map,
        const bool is_gather_rank
    );

    /**
    * \brief This collective function gathers data generated with compute_package_for_gather_rank
    * to the gather rank.
    * If my_rank != gather_rank the function returns dummy data. Otherwise the function returns
    * a pair containing:
    * 1) a byte array containing info on messages seen by other ranks
    * 2) a vector of displacements to access data corresponding to a given rank
    *
    * @param[in] package_for_gather_rank a byte array generated by compute_package_for_gather_rank
    * @param[in] gather_rank the ID of the "gather rank"
    * @param[in] my_rank the ID of the current rank
    * @return (see function description)
    */
    std::pair<std::vector<char>, std::vector<int>>
    gather_all_data(
        const std::vector<char>& package_for_gather_rank,
        const int gather_rank, const int my_rank);

    /**
    * \brief This function converts a vector of Msg struct into a byte array
    *
    * @param[in] msgs the vector of Msg struct
    * @return a byte array
    */
    std::vector<char> serialize_msgs(
        const std::vector<Msg>& msgs);

    /**
    * \brief This function converts a byte array into a vector of Msg struct
    *
    * @param[in] serialized the byte array
    * @return a vector of Msg struct
    */
    std::vector<Msg> deserialize_msgs(
        const std::vector<char>& serialized);
}
#endif

std::string abl_msg_logger::PriorityToString(const Priority& priority)
{
    if(priority == Priority::high)
        return "high";
    else if (priority == Priority::medium)
        return "medium";
    else
        return "low";
}

Priority abl_msg_logger::StringToPriority(const std::string& priority_string)
{
    if(priority_string == "high")
        return Priority::high;
    else if (priority_string == "medium")
        return Priority::medium;
    else if (priority_string == "low")
        return Priority::low;
    else
        ABLASTR_ABORT_WITH_MESSAGE(
            "Priority string '" + priority_string + "' not recognized");

    //this silences a "non-void function does not return a value in all control paths" warning
    return Priority::low;
}

std::vector<char> Msg::serialize() const
{
    std::vector<char> serialized_msg;

    abl_ser::put_in(this->topic, serialized_msg);
    abl_ser::put_in(this->text, serialized_msg);
    const int int_priority = static_cast<int>(this->priority);
    abl_ser::put_in(int_priority, serialized_msg);

    return serialized_msg;
}

Msg Msg::deserialize (std::vector<char>::const_iterator& it)
{
    Msg msg;

    msg.topic = abl_ser::get_out<std::string> (it);
    msg.text = abl_ser::get_out<std::string> (it);
    msg.priority = static_cast<Priority> (abl_ser::get_out<int> (it));

    return msg;
}

Msg Msg::deserialize (std::vector<char>::const_iterator&& it)
{
    return Msg::deserialize(it);
}

std::vector<char> MsgWithCounter::serialize() const
{
    std::vector<char> serialized_msg_with_counter;

    abl_ser::put_in_vec(msg.serialize(), serialized_msg_with_counter);
    abl_ser::put_in(this->counter, serialized_msg_with_counter);

    return serialized_msg_with_counter;
}

MsgWithCounter MsgWithCounter::deserialize (std::vector<char>::const_iterator& it)
{
    MsgWithCounter msg_with_counter;

    const auto vec = abl_ser::get_out_vec<char>(it);
    auto iit = vec.begin();
    msg_with_counter.msg = Msg::deserialize(iit);
    msg_with_counter.counter = abl_ser::get_out<std::int64_t> (it);

    return msg_with_counter;
}

MsgWithCounter MsgWithCounter::deserialize (std::vector<char>::const_iterator&& it)
{
    return MsgWithCounter::deserialize(it);
}

std::vector<char> MsgWithCounterAndRanks::serialize() const
{
    std::vector<char> serialized_msg_with_counter_and_ranks;

    abl_ser::put_in_vec(this->msg_with_counter.serialize(), serialized_msg_with_counter_and_ranks);
    abl_ser::put_in(this->all_ranks, serialized_msg_with_counter_and_ranks);
    abl_ser::put_in_vec(this->ranks, serialized_msg_with_counter_and_ranks);

    return serialized_msg_with_counter_and_ranks;
}

MsgWithCounterAndRanks
MsgWithCounterAndRanks::deserialize (std::vector<char>::const_iterator& it)
{
    MsgWithCounterAndRanks msg_with_counter_and_ranks;

    const auto vec = abl_ser::get_out_vec<char>(it);
    auto iit = vec.begin();
    msg_with_counter_and_ranks.msg_with_counter = MsgWithCounter::deserialize(iit);
    msg_with_counter_and_ranks.all_ranks = abl_ser::get_out<bool>(it);
    msg_with_counter_and_ranks.ranks = abl_ser::get_out_vec<int>(it);

    return msg_with_counter_and_ranks;
}

MsgWithCounterAndRanks
MsgWithCounterAndRanks::deserialize (std::vector<char>::const_iterator&& it)
{
    return MsgWithCounterAndRanks::deserialize(it);
}

Logger::Logger() :
    m_rank{amrex::ParallelDescriptor::MyProc()},
    m_num_procs{amrex::ParallelDescriptor::NProcs()},
    m_io_rank{amrex::ParallelDescriptor::IOProcessorNumber()}
{}

void Logger::record_msg(Msg msg)
{
    m_messages[msg]++;
}

std::vector<Msg> Logger::get_msgs() const
{
    auto res = std::vector<Msg>{};

    for (const auto& msg_w_counter : m_messages)
        res.emplace_back(msg_w_counter.first);

    return res;
}

std::vector<MsgWithCounter> Logger::get_msgs_with_counter() const
{
    auto res = std::vector<MsgWithCounter>{};

    for (const auto& msg : m_messages)
        res.emplace_back(MsgWithCounter{msg.first, msg.second});

    return res;
}

std::vector<MsgWithCounterAndRanks>
Logger::collective_gather_msgs_with_counter_and_ranks() const
{

#ifdef AMREX_USE_MPI

    // Trivial case of only one rank
    if (m_num_procs == 1)
        return one_rank_gather_msgs_with_counter_and_ranks();

    // Find out who is the "gather rank" and how many messages it has
    const auto my_msgs = get_msgs();
    const auto how_many_msgs = static_cast<int>(my_msgs.size());
    const auto [gather_rank, gather_rank_how_many_msgs] =
        find_gather_rank_and_its_msgs(how_many_msgs);

    // If the "gather rank" has zero messages there are no messages at all
    if(gather_rank_how_many_msgs == 0)
        return std::vector<MsgWithCounterAndRanks>{};

    // All the ranks receive the msgs of the "gather rank" as a byte array
    const auto serialized_gather_rank_msgs =
        ::get_serialized_gather_rank_msgs(my_msgs, gather_rank, m_rank);

    // Each rank assembles a message to send back to the "gather rank"
    const bool is_gather_rank = (gather_rank == m_rank);
    const auto package_for_gather_rank =
        ::compute_package_for_gather_rank(
            serialized_gather_rank_msgs,
            gather_rank_how_many_msgs,
            m_messages, is_gather_rank);

    // Send back all the data to the "gather rank"
    const auto [all_data, displacements] =
        ::gather_all_data(
            package_for_gather_rank,
            gather_rank, m_rank);

    // Use the gathered data to generate (on the "gather rank") a vector of all the
    // messages seen by all the ranks with the corresponding counters and
    // emitting rank lists.
    auto msgs_with_counter_and_ranks =
        compute_msgs_with_counter_and_ranks(
            m_messages,
            all_data,
            displacements,
            gather_rank);

    // If the current rank is not the I/O rank, send msgs_with_counter_and_ranks
    // to the I/O rank
    swap_with_io_rank(
        msgs_with_counter_and_ranks,
        gather_rank);

    return msgs_with_counter_and_ranks;
#else
    return one_rank_gather_msgs_with_counter_and_ranks();
#endif
}

std::vector<MsgWithCounterAndRanks>
Logger::one_rank_gather_msgs_with_counter_and_ranks() const
{
    std::vector<MsgWithCounterAndRanks> res;
    for (const auto& el : m_messages)
    {
        res.emplace_back(
            MsgWithCounterAndRanks{
                MsgWithCounter{el.first, el.second},
                true,
                std::vector<int>{m_rank}});
    }
    return res;
}

#ifdef AMREX_USE_MPI

std::pair<int,int> Logger::find_gather_rank_and_its_msgs(int how_many_msgs) const
{
    int max_items = 0;
    int max_rank = 0;

    const auto num_msg =
        amrex::ParallelDescriptor::Gather(how_many_msgs, m_io_rank);

    const auto m_am_i_io = (m_rank == m_io_rank);
    if (m_am_i_io){
        const auto it_max = std::max_element(num_msg.begin(), num_msg.end());
        max_items = *it_max;

        //In case of an "ex aequo" the I/O rank should be the gather rank
        max_rank = (max_items == how_many_msgs) ?
            m_io_rank : static_cast<int>(it_max - num_msg.begin());
    }

    auto package = std::array<int,2>{max_rank, max_items};
    amrex::ParallelDescriptor::Bcast(package.data(), 2, m_io_rank);

    return std::make_pair(package[0], package[1]);
}

std::vector<MsgWithCounterAndRanks>
Logger::compute_msgs_with_counter_and_ranks(
    const std::map<Msg,std::int64_t>& my_msg_map,
    const std::vector<char>& all_data,
    const std::vector<int>& displacements,
    const int gather_rank) const
{
    if(m_rank != gather_rank) return std::vector<MsgWithCounterAndRanks>{};

    std::vector<MsgWithCounterAndRanks> msgs_with_counter_and_ranks;

    // Put messages of the gather rank in msgs_with_counter_and_ranks
    msgs_with_counter_and_ranks.reserve(my_msg_map.size());
    for (const auto& el : my_msg_map)
    {
        msgs_with_counter_and_ranks.emplace_back(
            MsgWithCounterAndRanks{
                MsgWithCounter{el.first, el.second},
                false,
                std::vector<int>{m_rank}});
    }

    // We need a temporary map
    std::map<Msg, MsgWithCounterAndRanks> tmap;

#ifdef AMREX_USE_OMP
    #pragma omp parallel for
#endif
    for(int rr = 0; rr < m_num_procs; ++rr){ //for each rank
        if(rr == gather_rank) // (skip gather_rank)
            continue;

        // get counters generated by rank rr
        auto it = all_data.begin() + displacements[rr];
        const auto counters_rr = abl_ser::get_out_vec<std::int64_t>(it);

         //for each counter from rank rr
        std::int64_t c = 0;
        for (const auto& counter : counters_rr){
#ifdef AMREX_USE_OMP
            #pragma omp atomic
#endif
            msgs_with_counter_and_ranks[c].msg_with_counter.counter +=
                counter; //update corresponding global counter

            //and add rank to rank list if it has emitted the message
            if (counter > 0){
#ifdef AMREX_USE_OMP
            #pragma omp critical
#endif
                {
                    msgs_with_counter_and_ranks[c].ranks.push_back(rr);
                }
            }
            c++;
        }

        // for each additional message coming from rank rr
        const auto how_many_additional_msgs_with_counter = abl_ser::get_out<int>(it);
        for(int i = 0; i < how_many_additional_msgs_with_counter; ++i){

            //deserialize the message
            const auto serialized_msg_with_counter = abl_ser::get_out_vec<char>(it);
            auto msg_with_counter =
                MsgWithCounter::deserialize(serialized_msg_with_counter.begin());

            //and eventually add it to the temporary map
#ifdef AMREX_USE_OMP
            #pragma omp critical
#endif
            {
                if (tmap.find(msg_with_counter.msg) == tmap.end()){
                    const auto msg_with_counter_and_ranks =
                        MsgWithCounterAndRanks{
                            msg_with_counter,
                            false,
                            std::vector<int>{rr}
                        };
                    tmap[msg_with_counter.msg] = msg_with_counter_and_ranks;
                }
                else{
                    tmap[msg_with_counter.msg].msg_with_counter.counter +=
                        msg_with_counter.counter;
                    tmap[msg_with_counter.msg].ranks.push_back(rr);
                }
            }
        }
    }

    // Check if messages emitted by "gather rank" are actually emitted by all ranks
    const auto ssize = static_cast<int>(msgs_with_counter_and_ranks.size());
    for (int i = 0; i < ssize; ++i){
        const auto how_many =
            static_cast<int>(msgs_with_counter_and_ranks[i].ranks.size());
        if(how_many == m_num_procs){
            msgs_with_counter_and_ranks[i].all_ranks = true;
            // trick to force free memory
            std::vector<int>{}.swap(msgs_with_counter_and_ranks[i].ranks);
        }
    }

    // Add elements from the temporary map
    for(const auto& el : tmap){
        msgs_with_counter_and_ranks.push_back(el.second);
    }

    // Sort affected ranks lists
    for(auto& el : msgs_with_counter_and_ranks){
        std::sort(el.ranks.begin(), el.ranks.end());
    }

    return msgs_with_counter_and_ranks;
}

void Logger::swap_with_io_rank(
    std::vector<MsgWithCounterAndRanks>& msgs_with_counter_and_ranks,
    int gather_rank) const
{
    if (gather_rank != m_io_rank){
        if(m_rank == gather_rank){
            auto package = std::vector<char>{};
            for (const auto& el: msgs_with_counter_and_ranks)
                abl_ser::put_in_vec<char>(el.serialize(), package);

            auto package_size = static_cast<int>(package.size());
            amrex::ParallelDescriptor::Send(&package_size, 1, m_io_rank, 0);
            amrex::ParallelDescriptor::Send(package, m_io_rank, 1);
            const auto list_size = static_cast<int>(msgs_with_counter_and_ranks.size());
            amrex::ParallelDescriptor::Send(&list_size, 1, m_io_rank, 2);
        }
        else if (m_rank == m_io_rank){
            int vec_size = 0;
            amrex::ParallelDescriptor::Recv(&vec_size, 1, gather_rank, 0);
            std::vector<char> package(vec_size);
            amrex::ParallelDescriptor::Recv(package, gather_rank, 1);
            int list_size = 0;
            amrex::ParallelDescriptor::Recv(&list_size, 1, gather_rank, 2);
            auto it = package.cbegin();
            for (int i = 0; i < list_size; ++i){
                const auto vec = abl_ser::get_out_vec<char>(it);
                msgs_with_counter_and_ranks.emplace_back(
                    MsgWithCounterAndRanks::deserialize(vec.begin())
                );
            }
        }
    }
}

namespace
{
std::vector<char>
get_serialized_gather_rank_msgs(
    const std::vector<Msg>& my_msgs,
    const int gather_rank,
    const int my_rank)
{
    const bool is_gather_rank = (my_rank == gather_rank);

    auto serialized_gather_rank_msgs = std::vector<char>{};
    int size_serialized_gather_rank_msgs = 0;

    if (is_gather_rank){
        serialized_gather_rank_msgs = ::serialize_msgs(my_msgs);
        size_serialized_gather_rank_msgs = static_cast<int>(
            serialized_gather_rank_msgs.size());
    }

    amrex::ParallelDescriptor::Bcast(
        &size_serialized_gather_rank_msgs, 1, gather_rank);

    if (!is_gather_rank)
        serialized_gather_rank_msgs.resize(
            size_serialized_gather_rank_msgs);

    amrex::ParallelDescriptor::Bcast(
        serialized_gather_rank_msgs.data(),
        size_serialized_gather_rank_msgs, gather_rank);

    return serialized_gather_rank_msgs;
}

std::vector<char>
compute_package_for_gather_rank(
    const std::vector<char>& serialized_gather_rank_msgs,
    const std::int64_t gather_rank_how_many_msgs,
    const std::map<Msg, std::int64_t>& my_msg_map,
    const bool is_gather_rank)
{
    if(!is_gather_rank){
        auto package = std::vector<char>{};

        //generates a copy of the message map
        auto msgs_to_send = std::map<Msg, std::int64_t>{my_msg_map};

        // For each message of the "gather rank" store how many times
        // the message has been emitted by the current ranks.
        const auto gather_rank_msgs =
            ::deserialize_msgs(serialized_gather_rank_msgs);
        std::vector<std::int64_t> gather_rank_msg_counters(gather_rank_how_many_msgs);
        std::int64_t counter = 0;
        for (const auto& msg : gather_rank_msgs){
            const auto pp = msgs_to_send.find(msg);
            if (pp != msgs_to_send.end()){
                gather_rank_msg_counters[counter] += pp->second;
                // Remove messages already seen by "gather rank" from
                // the messages to send back
                msgs_to_send.erase(msg);
            }
            counter++;
        }
        abl_ser::put_in_vec(gather_rank_msg_counters, package);

        // Add the additional messages seen by the current rank to the package
        abl_ser::put_in(static_cast<int>(msgs_to_send.size()), package);
        for (const auto& el : msgs_to_send)
            abl_ser::put_in_vec<char>(
                MsgWithCounter{el.first, el.second}.serialize(), package);

        return package;
    }

    return std::vector<char>{};
}

std::pair<std::vector<char>, std::vector<int>>
gather_all_data(
    const std::vector<char>& package_for_gather_rank,
    const int gather_rank, const int my_rank)
{
    auto package_lengths = std::vector<int>{};
    auto all_data = std::vector<char>{};
    auto displacements = std::vector<int>{};

    if(gather_rank != my_rank){
        amrex::ParallelDescriptor::Gather(
            static_cast<int>(package_for_gather_rank.size()), gather_rank);
        amrex::ParallelDescriptor::Gatherv(
            package_for_gather_rank.data(),
            static_cast<int>(package_for_gather_rank.size()),
            all_data.data(),
            package_lengths,
            displacements,
            gather_rank);
    }
    else{
        const int zero_size = 0;
        package_lengths =
            amrex::ParallelDescriptor::Gather(zero_size, gather_rank);

        // Compute displacements
        // Given (n1, n2, n3, n4, ..., n_n) we need (0, n1, n1+n2, n1+n2+n3, ...),
        // but partial_sum gives us (n1,n1+n2, n1+n2+n3, n1+n2+n3+n4, ...).
        // Rotating this last vector by one is just shifting: (n1+n2+n3+n4+...,n1, n1+n2, n1+n2+n3, ...).
        // Then we just need to replace the first element with zero: (0,n1, n1+n2, n1+n2+n3, ...).
        displacements.resize(package_lengths.size());
        std::partial_sum(package_lengths.begin(), package_lengths.end(),
            displacements.begin());
        const auto total_sum = displacements.back();
        std::rotate(displacements.rbegin(),
            displacements.rbegin()+1,
            displacements.rend());
        displacements[0] = 0;

        all_data.resize(total_sum);

        amrex::ParallelDescriptor::Gatherv(
            static_cast<char*>(nullptr),
            0,
            all_data.data(),
            package_lengths,
            displacements,
            gather_rank);
    }
    return std::make_pair(all_data, displacements);
}

std::vector<char> serialize_msgs(
    const std::vector<Msg>& msgs)
{
    auto serialized = std::vector<char>{};

    const auto how_many = static_cast<int> (msgs.size());
    abl_ser::put_in (how_many, serialized);

    for (const auto& msg : msgs){
        abl_ser::put_in_vec(msg.serialize(), serialized);
    }
    return serialized;
}

std::vector<Msg> deserialize_msgs(
    const std::vector<char>& serialized)
{
    auto it = serialized.begin();

    const auto how_many = abl_ser::get_out<int>(it);
    auto msgs = std::vector<Msg>{};
    msgs.reserve(how_many);

    for (int i = 0; i < how_many; ++i){
        const auto vv = abl_ser::get_out_vec<char>(it);
        msgs.emplace_back(Msg::deserialize(vv.begin()));
    }

    return msgs;
}
}

#endif
