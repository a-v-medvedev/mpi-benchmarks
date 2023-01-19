#pragma once

#include <sstream>
#include <sys/time.h>

struct timer {
    const time_t sec_to_usec = 1000000L;
    std::string name;
    bool do_out = false;
    bool stopped = false;
    std::stringstream comment;
    timeval tv[2];
    long *presult = nullptr;
    timer(const std::string& _name = "", bool _do_out = false) :
        name(_name), do_out(_do_out) {
            gettimeofday(&tv[0], NULL);
        }
    long time_diff() {
        return ((long)tv[1].tv_sec - (long)tv[0].tv_sec) * sec_to_usec + (long)tv[1].tv_usec - (long)tv[0].tv_usec;
    }
    timer(long *_presult) : presult(_presult) {
        gettimeofday(&tv[0], NULL);
    }
    long stop() {
        gettimeofday(&tv[1], NULL);
        long diff = time_diff();
        if (presult) {
            *presult = diff;
            return diff;
        }
        if (do_out) {
            std::cout << name << ": " << "[ " << "time (usec): " << diff;
            if (comment.str().size()) {
                std::cout << ", " << "comment: \"" << comment.str() << "\"";
            }
            std::cout << " " << "]" << std::endl;
        }
        stopped = true;
        return diff;
    }
    ~timer() throw() {
        if (!stopped)
            stop();
    }
};

