#include <iostream>
struct commactype : std::ctype<char> {
    commactype() : std::ctype<char>(get_table()) {}
    std::ctype_base::mask const* get_table(){
        std::ctype_base::mask* rc = 0;
        if (rc == 0){
            rc = new std::ctype_base::mask[std::ctype<char>::table_size];
            std::fill_n(rc, std::ctype<char>::table_size, std::ctype_base::mask());
            rc[','] = std::ctype_base::space;
            rc[' '] = std::ctype_base::space;
            rc['\t'] = std::ctype_base::space;
            rc['\n'] = std::ctype_base::space;
            rc['\r'] = std::ctype_base::space;
        }
        return rc;
    }
};