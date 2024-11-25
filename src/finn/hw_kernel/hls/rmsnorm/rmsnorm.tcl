set top rmsnorm 
open_project $top.hls
add_files ${top}.cpp -cflags "-std=c++14 -Wall -O2"
add_files -tb ${top}_tb.cpp -cflags "-std=c++14 -Wall -O2"
set_top ${top}
open_solution sol1
#set_part xczu3eg-sbva484-1-i
set_part xcv80-lsva4737-2MHP-e-S
create_clock -period 5 -name default
csim_design
csynth_design -dump_post_cfg
cosim_design

#export_design -description "BERT Softmax" -display_name "Softmax" -format ip_catalog -ipname softmax -vendor "amd.com"

exit
