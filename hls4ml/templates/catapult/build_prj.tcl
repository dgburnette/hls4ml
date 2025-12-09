#################
#    HLS4ML
#################

# Insert BuildOptions - default values come from config
# but may be overridden in hlsmodel.build() args
#hls-fpga-machine-learning insert build_options

# Get pathname to this script to use as dereference path for relative file pathnames
set sfd [file dirname [info script]]

if { [info exists ::argv] } {
  foreach arg $::argv {
    foreach {optname optval} [split $arg '='] {}
    # map depricated option names to new names
    set mapping {"cosim" "SCVerify" "validation" "SCVerify" "synth" "Synth" "vsynth" "RTLSynth" "ran_frame" "RandomTBFrames" "sw_opt" "PowerEst" "power" "PowerOpt" "da" "LaunchDA" "bup" "BuildBUP"}
    set pos [lsearch -exact $mapping $optname]
    if { ($pos != -1) && ([expr $pos % 2] == 0) } {
      set oldoptname $optname
      set optname [lindex $mapping [expr $pos + 1]]
      logfile message "HLS4ML build() option '$oldoptname' is being depricated. Use '$optname'\n" warning
    }
    if { [info exists BuildOptions($optname)] } {
      if {[string is integer -strict $optval]} {
        set BuildOptions($optname) $optval
      } else {
        set BuildOptions($optname) [string is true -strict $optval]
      }
    } else {
      logfile message "Unknown argv switch '$optname'\n" error
    }
  }
}

puts "***** INVOKE OPTIONS *****"
foreach x [lsort [array names BuildOptions]] {
  puts "[format {   %-20s %s} $x $BuildOptions($x)]"
}
puts ""

proc report_time { op_name time_start time_end } {
  set time_taken [expr $time_end - $time_start]
  set time_s [expr ($time_taken / 1000) % 60]
  set time_m [expr ($time_taken / (1000*60)) % 60]
  set time_h [expr ($time_taken / (1000*60*60)) % 24]
  puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

proc setup_xilinx_part { part } {
  global env

  # Map Xilinx PART into Catapult library names
  set part_sav $part
  set libname [lindex [library get /CONFIG/PARAMETERS/Vivado/PARAMETERS/Xilinx/PARAMETERS/*/PARAMETERS/*/PARAMETERS/$part/LIBRARIES/*/NAME -match glob -ret v] 0]
  puts "Library Name: $libname"
  if { [llength $libname] == 1 } {
    set libpath [library get /CONFIG/PARAMETERS/Vivado/PARAMETERS/Xilinx/PARAMETERS/*/PARAMETERS/*/PARAMETERS/$part/LIBRARIES/*/NAME -match glob -ret p]
    puts "Library Path: $libpath"
    if { [regexp {/CONFIG/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/.*} $libpath dummy rtltool vendor family speed part] } {
      solution library add $libname -- -rtlsyntool $rtltool -vendor $vendor -family $family -speed $speed -part $part_sav
    } else {
      solution library add $libname -- -rtlsyntool Vivado
    }
  } else {
    logfile message "Could not find specific Xilinx base library for part '$part'. Using KINTEX-u\n" warning
    solution library add mgc_Xilinx-KINTEX-u-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family KINTEX-u -speed -2 -part xcku115-flvb2104-2-i
  }
  solution library add Xilinx_RAMS
  solution library add Xilinx_ROMS
  solution library add Xilinx_FIFO
  # Point to AMD/Xilinx/Vivado precompiled library cache  
  if { [info exists env(XILINX_PCL_CACHE)] } {
    options set /Flows/Vivado/PCL_CACHE $env(XILINX_PCL_CACHE)
    solution options set /Flows/Vivado/PCL_CACHE $env(XILINX_PCL_CACHE)
  }
}


proc setup_asic_libs { args } {
  global env
  set do_saed 0
  foreach lib $args {
    solution library add $lib -- -rtlsyntool DesignCompiler
    if { [lsearch -exact {saed32hvt_tt0p78v125c_beh saed32lvt_tt0p78v125c_beh saed32rvt_tt0p78v125c_beh} $lib] != -1 } {
      set do_saed 1
      # Special case for SAED32 and ADVDPOPT (taken from setup_saed.tcl)
      if { [info exists env(SAED32_EDK)] } {
        # Technology Process for layout
        options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) tech] -append
        # Design Compiler Library (DBs for cells and wire loads)
        options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) lib SG db] -append
        ## Prevents $MGC_HOME/pkgs/siflibs/saed from appearing in the cache header for OnTheFly
        #options set ComponentLibs/TechLibSearchPath [file join $::env(MGC_HOME) pkgs siflibs saed ] -append
        options set Flows/DesignCompiler/EnableWireloadSettings true
        #options set Flows/DesignCompiler/libs_db [solution get /TECHLIBS/$lib/VARS/libs_db/VALUE]
        # MilkyWay Libraries (cell, tech, TLU, and map files)
        options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) tech milkyway] -append
        options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) lib stdcell_rvt milkyway] -append
        options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) lib stdcell_hvt milkyway] -append
        options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) lib stdcell_lvt milkyway] -append
        # Liberty and LEF Libraries
        options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) lib SG] -append
        #options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) lib stdcell_${vth} lef] -append
        # TLU files
        options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) tech star_rcxt] -append
        #options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) lib SG] -append
      }
    }
  }
  solution library add ccs_sample_mem
  solution library add ccs_sample_rom
  solution library add hls4ml_lib
  #hls-fpga-machine-learning insert directives libraries
  go libraries

  # special exception for SAED32 for use in power estimation
  if { $do_saed } {
    # SAED32 selected - enable DC settings to access Liberty data for power estimation
    source [application get /SYSTEM/ENV_MGC_HOME]/pkgs/siflibs/saed/setup_saedlib.tcl
  }
}

options set Input/CppStandard {c++17}
options set Input/CompilerFlags -DRANDOM_FRAMES=$BuildOptions(RandomTBFrames)
options set Input/SearchPath {$MGC_HOME/shared/include/nnet_utils} -append
options set ComponentLibs/SearchPath {$MGC_HOME/shared/pkgs/ccs_hls4ml} -append

if {$BuildOptions(reset)} {
  project load CATAPULT_DIR.ccs
  go new
} else {
  project new -name CATAPULT_DIR
}

#--------------------------------------------------------
# Configure Catapult Options
# downgrade HIER-10
options set Message/ErrorOverride HIER-10 -remove
solution options set Message/ErrorOverride HIER-10 -remove
options set Message/Hide HIER-10 -append

if {$BuildOptions(vhdl)}    {
  options set Output/OutputVHDL true
} else {
  options set Output/OutputVHDL false
}
if {$BuildOptions(verilog)} {
  options set Output/OutputVerilog true
} else {
  options set Output/OutputVerilog false
}

#--------------------------------------------------------
# Configure Catapult Flows
if { [info exists ::env(XILINX_PCL_CACHE)] } {
options set /Flows/Vivado/PCL_CACHE $::env(XILINX_PCL_CACHE)
solution options set /Flows/Vivado/PCL_CACHE $::env(XILINX_PCL_CACHE)
}

#--------------------------------------------------------
# Source optional end-user startup script
if { [info exists BuildOptions(startup)] && ($BuildOptions(startup) != "") && [file exists $BuildOptions(startup)] } {
  logfile message "Sourcing Catapult AI NN startup script '$BuildOptions(startup)'\n" info
  catch {dofile $BuildOptions(startup)}
}

# Turn on HLS4ML flow (wrapped in a cache so that older Catapult installs still work)
catch {flow package require /HLS4ML}

# Turn on SCVerify flow
flow package require /SCVerify
#  flow package option set /SCVerify/INVOKE_ARGS {$sfd/firmware/weights $sfd/tb_data/tb_input_features.dat $sfd/tb_data/tb_output_predictions.dat}
#hls-fpga-machine-learning insert invoke_args

# Turn on VSCode flow
# flow package require /VSCode
# To launch VSCode on the C++ HLS design:
#   cd my-Catapult-test
#   code Catapult.code-workspace

#--------------------------------------------------------
#    Start of HLS script
set design_top myproject
solution file add $sfd/firmware/myproject.cpp
solution file add $sfd/myproject_test.cpp -exclude true

# Parse hls4ml_config.yml
set Strategy latency
set IOType io_stream
set ac_sync_workaround 0
if { ![file exists $sfd/hls4ml_config.yml] } {
  logfile message "Could not locate HLS4ML configuration file '$sfd/hls4ml_config.yml'. Unable to determine network configuration.\n" warning
} else {
  set pf [open "$sfd/hls4ml_config.yml" "r"]
  while {![eof $pf]} {
    gets $pf line
    if { [regexp {\s+Strategy: (\w+)} $line all value] }     { set Strategy [string tolower $value] }
    if { [regexp {IOType: (\w+)} $line all value] }          { set IOType [string tolower $value] }
    if { [regexp {ParamStore: merged} $line all value] }     { set ac_sync_workaround 1 }
  }
  close $pf
}
if { $ac_sync_workaround } {
  directive set -STRICT_MIO_SCHEDULING false
  directive set -STRICT_WAITSYNC_IO_SCHEDULING false
}

if { $IOType == "io_stream" } {
solution options set Architectural/DefaultRegisterThreshold 2050
}
directive set -RESET_CLEARS_ALL_REGS no
# Constrain arrays to map to memory only over a certain size
directive set -MEM_MAP_THRESHOLD [expr 2048 * 16 + 1]
directive set -REDUNDANT_RTL_ELIMINATION false
directive set -ADVANCED_DPOPT false
# The following line gets modified by the backend writer
set hls_clock_period 5

#hls-fpga-machine-learning insert directives analyze
go analyze

# Workaround for io_parallel and separable conv2d
if { $IOType == "io_parallel" } {
  set inlines {}
  set pooling2d_used 0
  set pooling1d_used 0
  foreach fn [solution get /SOURCEHIER/FUNC_HBS/* -match glob -ret l -checkpath 0] {
    if { [string match {nnet::pooling2d_cl*} $fn] } { set pooling2d_used 1 }
    if { [string match {nnet::pooling1d_cl*} $fn] } { set pooling1d_used 1 }
    if { [string match {ac::fx_div*} $fn] }         { lappend inlines $fn }
  }
  foreach fn $inlines {
    set old [solution design get $fn]
    logfile message "solution design set $fn -inline\n" warning
    solution design set $fn -inline
  }
  if { $pooling2d_used || $pooling1d_used} {
    # Need to enable this since pooling?d_cl has some division operations in it
    directive set -SCHED_USE_MULTICYCLE true
  }
}

# NORMAL TOP DOWN FLOW
if { ! $BuildOptions(BuildBUP) } {

#hls-fpga-machine-learning insert directives compile
go compile

if {$BuildOptions(csim)} {
  puts "***** C SIMULATION *****"
  set time_start [clock clicks -milliseconds]
  flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
  set time_end [clock clicks -milliseconds]
  report_time "C SIMULATION" $time_start $time_end
}

puts "***** SETTING TECHNOLOGY LIBRARIES *****"
#hls-fpga-machine-learning insert techlibs

directive set -CLOCKS [list clk [list -CLOCK_PERIOD $hls_clock_period -CLOCK_EDGE rising -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high]]

if {$BuildOptions(Synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]

  #hls-fpga-machine-learning insert directives assembly

  go assembly

  #hls-fpga-machine-learning insert directives architect

  go architect

  #hls-fpga-machine-learning insert directives allocate

  go allocate

  #hls-fpga-machine-learning insert directives schedule

  go schedule

  #hls-fpga-machine-learning insert directives extract

  go extract
  set time_end [clock clicks -milliseconds]
  report_time "C/RTL SYNTHESIS" $time_start $time_end
}

# BOTTOM-UP FLOW
} else {
  # Start at 'go analyze'
  go analyze

  # Build the design bottom-up
  directive set -CLOCKS [list clk [list -CLOCK_PERIOD $hls_clock_period -CLOCK_EDGE rising -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high]]

  set blocks [solution get /HIERCONFIG/USER_HBS/*/RESOLVED_NAME -match glob -rec 1 -ret v -state analyze]
  set bu_mappings {}
  set top [lindex $blocks 0]
  foreach block [lreverse [lrange $blocks 1 end]] {
    # skip blocks that are net nnet:: functions
    if { [string match {nnet::*} $block] == 0 } { continue }
    go analyze
    solution design set $block -top
    go compile
    solution library remove *
    puts "***** SETTING TECHNOLOGY LIBRARIES *****"
#hls-fpga-machine-learning insert techlibs
    go extract
    set block_soln "[solution get /TOP/name -checkpath 0].[solution get /VERSION -checkpath 0]"
    lappend bu_mappings [solution get /CAT_DIR] /$top/$block "\[Block\] $block_soln"
  }

  # Move to top design
  go analyze
  solution design set $top -top
  go compile

  if {$BuildOptions(csim)} {
    puts "***** C SIMULATION *****"
    set time_start [clock clicks -milliseconds]
    flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
    set time_end [clock clicks -milliseconds]
    report_time "C SIMULATION" $time_start $time_end
  }
  foreach {d i l} $bu_mappings {
    logfile message "solution options set ComponentLibs/SearchPath $d -append\n" info
    solution options set ComponentLibs/SearchPath $d -append
  }

  # Add bottom-up blocks
  puts "***** SETTING TECHNOLOGY LIBRARIES *****"
  solution library remove *
#hls-fpga-machine-learning insert techlibs
  # need to revert back to go compile
  go compile
  foreach {d i l} $bu_mappings {
    logfile message "solution library add [list $l]\n" info
    eval solution library add [list $l]
  }
  go libraries

  # Map to bottom-up blocks
  foreach {d i l} $bu_mappings {
    # Make sure block exists
    set cnt [directive get $i/* -match glob -checkpath 0 -ret p]
    if { $cnt != {} } {
      logfile message "directive set $i -MAP_TO_MODULE [list $l]\n" info
      eval directive set $i -MAP_TO_MODULE [list $l]
    }
  }
  go assembly
  set design [solution get -name]
  logfile message "Adjusting FIFO_DEPTH for top-level interconnect channels\n" warning
  # FIFO interconnect between layers
  foreach ch_fifo_m2m [directive get -match glob -checkpath 0 -ret p $design/*_out:cns/MAP_TO_MODULE] {
    set ch_fifo [join [lrange [split $ch_fifo_m2m '/'] 0 end-1] /]/FIFO_DEPTH
    logfile message "directive set -match glob $ch_fifo 1\n" info
    directive set -match glob "$ch_fifo" 1
  }
  # For bypass paths - the depth will likely need to be larger than 1
  foreach ch_fifo_m2m [directive get -match glob -checkpath 0 -ret p $design/*_cpy*:cns/MAP_TO_MODULE] {
    set ch_fifo [join [lrange [split $ch_fifo_m2m '/'] 0 end-1] /]/FIFO_DEPTH
    logfile message "Bypass FIFO '$ch_fifo' depth set to 1 - larger value may be required to prevent deadlock\n" warning
    logfile message "directive set -match glob $ch_fifo 1\n" info
    directive set -match glob "$ch_fifo" 1
  }
  go architect
  go allocate
  go schedule
  go dpfsm
  go extract
}

project save

if {$BuildOptions(SCVerify) } {
  if {$BuildOptions(verilog)} {
    flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim
  }
  if {$BuildOptions(vhdl)} {
    flow run /SCVerify/launch_make ./scverify/Verify_rtl_vhdl_msim.mk {} SIMTOOL=msim sim
  }
}

if {$BuildOptions(PowerEst)} {
  puts "***** Pre Power Optimization *****"
  go switching
  if {$BuildOptions(verilog)} {
    flow run /PowerAnalysis/report_pre_pwropt_Verilog
  }
  if {$BuildOptions(vhdl)} {
    flow run /PowerAnalysis/report_pre_pwropt_VHDL
  }
}

if {$BuildOptions(PowerOpt)} {
  puts "***** Power Optimization *****"
  go power
}

if {$BuildOptions(RTLSynth)} {
  # find last RTL synthesis script created (either extract or power stage)
  set launch {}
  foreach {p v} [solution get /OUTPUTFILES/.../FILETYPE -match glob -rec 1 -ret pv] {
    if { $v == "SYNTHESIS" } {
      set p1 [file dirname $p]
      set nettype [string tolower [solution get $p1/DEPENDENCIES/1/FILETYPE -checkpath 0]]
      if { [info exists BuildOptions($nettype)] && $BuildOptions($nettype) } {
        set launch [lindex [lindex [solution get $p1/FLOWS] 0] 1]
      }
    }
  }
  if { $launch != {} } {
    puts "***** RTL SYNTHESIS *****"
    set time_start [clock clicks -milliseconds]
    eval flow run $launch
    set time_end [clock clicks -milliseconds]
    report_time "RTL SYNTHESIS" $time_start $time_end
  } else {
    logfile message "RTL Synthesis flow lookup failed\n" warning
  }
}

if {$BuildOptions(LaunchDA)} {
  puts "***** Launching DA *****"
  flow run /DesignAnalyzer/launch
}

if { [catch {flow package present /HLS4ML}] == 0 } {
  flow run /HLS4ML/collect_reports
}
