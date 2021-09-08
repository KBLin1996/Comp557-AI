wm title . "PacWar Simulator"
wm geometry . +0+0


if { [winfo depth .] == "1" } {
    set color(0) "black"
    set color(spec1) "black"
    set color(1) "black"
    set color(spec2) "black"
} else {
    set color(0) "red"
    set color(spec1) "red"            
    set color(1) "blue"
    set color(spec2) "blue"
}

option add *Font "-Adobe-Helvetica-Bold-R-Normal--*-120-*"


# ------------------------------------------------
# --------------- user input constants -----------
set status reset     ;# status of the simulator
#                      "stop"   : simulator is halted
#                      "step"   : simulator is in single-step mode
#                      "run"    : simulator is running
#                      "reset " : simulator is reset
set size 16          ;# size of the bitmaps used in display: 16, 32 or 48
set showRounds 1     ;# display each round; o/w only when stopped
set kind duel        ;# how many kinds of mites to simulate at once
#                       "duel" is 2 species
#                       "test" is 1 species
set spec1name ones   ;# name of first species
set spec2name threes ;# name of second species (only used in duel)

set spec1Cnt 0       ;# how many species 1 mites are alive
set spec2Cnt 0       ;# how many species 2 mites are alive
set display_clockCount 0  ;# how long simulation has run

# These are the pre-loaded species
set spec(ones) "11111111111111111111111111111111111111111111111111"
set spec(threes) "33333333333333333333333333333333333333333333333333"

# ------------------------------------------------
# ---------------- utility procedures ------------

# This prepares a window, .$d, to be a modal dialog.
#
# The remaining args are the names of the other buttons 
# to be added (the default is OK).
#
# To use it, put the "body" of the dialog in .$d.top,
# display .$d, do a tkwait on ${d}_result, which will
# contain the number of the button pressed (1=ok, 2=...)
proc MyDialogPrep {d args} {
    global ${d}_result

    toplevel .$d

    frame .$d.top -relief raised -bd 1
    pack .$d.top -side top -fill both
    frame .$d.bot -relief raised -bd 1
    pack .$d.bot -side bottom -fill both

    frame .$d.bot.dflt -relief sunken -bd 1
    button .$d.bot.dflt.ok -text OK -command "set ${d}_result 1" \
	   -pady 2m -padx 2m
    pack .$d.bot.dflt.ok -expand 1
    pack .$d.bot.dflt -side left -ipadx 2m -ipady 1m -expand 1 -pady 2m
    set i 2
    foreach b $args {
    	button .$d.bot.b$b -text $b -command "set ${d}_result $i"
	incr i
    	pack .$d.bot.b$b -side left -ipadx 2m -ipady 1m -expand 1 -pady 2m
    }
    set ${d}_result 0
}

#
# Do a single-entry dialog, in window .$d, with window title $title, 
# entry-label $lbl, inital value $val, the width of the entry, and 
# use #scanPat to have scan validate the thing
# Return the new value, an empty string if Canceled
# To be nice, we've set things up so hitting Return is just like hitting OK
#
proc EntryDialog {d title lbl val width scanPat} {
    set resName [format "%s_result" $d]
    global $resName

    MyDialogPrep $d Cancel

    wm title .$d $title

    label .$d.lbl -text $lbl
    entry .$d.entry -width $width -relief sunken
    eval "bind .$d.entry <Return> {set $resName 1}"
    .$d.entry insert 0 $val
    pack .$d.lbl .$d.entry -in .$d.top -side left -ipadx 2m -ipady 1m \
        -expand 1 -pady 2m
	
    set oldFocus [focus]
    while { [set $resName] == 0 } {
       grab set .$d
       focus .$d
       tkwait variable $resName
       if { [set $resName] == 1 } {
	  if { [scan [.$d.entry get] $scanPat val] != 1 } {
	    tk_dialog .dlg "" "Invaid number" warning 0 OK
	    set $resName 0
	    set val ""
	  }  
       }
    }
    destroy .$d
    focus $oldFocus
    unset $resName
    return $val
}


# ------------------------------------------------
# ---------- display update procedures -----------

#
# Update cell (x,y) to show species s, direction d and age a.
# If s>1, show cell to be empty
#
proc DrawCell { x y s d a } {
    global size color
    if { $s > 1 } {
    	.arena.x${x}y$y configure -bitmap blob$size -fg black
    } else {
    	.arena.x${x}y$y configure -bitmap pac${size}_$s$d$a -fg $color($s)
    }
}

#
# Reset the display for a new round
#
proc ClearArena {} {
    global size spec1Cnt spec2Cnt display_clockCount

    set spec1Cnt 0
    set spec2Cnt 0
    set display_clockCount 0
    for {set y 1} {$y < 10} {incr y} {
	for {set x 1} {$x < 20} {incr x} {
	    .arena.x${x}y$y configure -bitmap blob$size -fg black
	}
    }
}

#
# Create the main display area
#
proc InitArena {} {
    global size

    for {set y 1} {$y < 10} {incr y} {
        frame .arena.r$y
	for {set x 1} {$x < 20} {incr x} {
	    label .arena.x${x}y$y -bitmap blob$size -bd 0
	    pack .arena.x${x}y$y -in .arena.r$y -side left
	}
    	pack .arena.r$y -side top
    }
    pack .arena
}


# ------------------------------------------------
# ------------ user entry procedures -------------

#
# Build up a widget w (really a frame) that allows the specification
# of a gene
#
proc MakeGeneWidget {w} {
    global $w chrom

    frame $w
    frame $w.left
    frame $w.right
    frame $w.u
    frame $w.v
    frame $w.vr0
    frame $w.vr1
    frame $w.vr2
    frame $w.vr3
    frame $w.w
    frame $w.x
    frame $w.y
    frame $w.yr0
    frame $w.yr1
    frame $w.yr2
    frame $w.yr3
    frame $w.z
    frame $w.zr0
    frame $w.zr1
    frame $w.zr2
    frame $w.zr3
    pack $w.u -in $w.left -side top -anchor nw
    pack $w.v -in $w.left -side top -anchor nw
    pack $w.w -in $w.left -side top -anchor nw
    pack $w.x -in $w.left -side top -anchor nw
    pack $w.y $w.z -in $w.right -side top
    pack $w.left $w.right -in $w -side left -anchor nw 
    for {set i 0} {$i < 50} {incr i} {
	set chrom($i) 0
	menubutton $w.g$i -textvariable chrom($i) -menu $w.g$i.m \
		-relief raised -width 1
	menu $w.g$i.m
	for {set j 0} {$j < 4} {incr j} { 
	    $w.g$i.m add radiobutton -variable chrom($i) \
	        -label $j -value $j -command "set chrom($i) $j"
	}
    }
    set i 0
    label $w.u.lbl -text U -width 2
    pack $w.u.lbl -side left
    for {} {$i < 4} {incr i} {
    	pack $w.g$i -in $w.u -side left
    }
    foreach j {0 1 2 3} {
	label $w.vr${j}.lbl -text "V$j" -width 2
	pack $w.vr${j}.lbl -side left
	foreach k {0 1 2 3} {
	    pack $w.g$i -in $w.vr$j -side left
	    incr i
	}
	pack $w.vr$j -in $w.v -side top
    }
    label $w.w.lbl -text W -width 2
    pack $w.w.lbl -side left
    foreach j {0 1 2} {
    	pack $w.g$i -in $w.w -side left
	incr i
    }
    label $w.x.lbl -text X -width 2
    pack $w.x.lbl -side left
    foreach j {0 1 2} {
    	pack $w.g$i -in $w.x -side left
	incr i
    }
    foreach j {0 1 2 3} {
	label $w.yr${j}.lbl -text "Y$j" -width 2
	pack $w.yr${j}.lbl -side left
	foreach k {0 1 2} {
	    pack $w.g$i -in $w.yr$j -side left
	    incr i
	}
	pack $w.yr$j -in $w.y -side top
    }
    foreach j {0 1 2 3} {
	label $w.zr${j}.lbl -text "Z$j" -width 2
	pack $w.zr${j}.lbl -side left
	foreach k {0 1 2} {
	    pack $w.g$i -in $w.zr$j -side left
	    incr i
	}
	pack $w.zr$j -in $w.z -side top
    }
}

#
# Build up a widget w (really a frame) for listing genes
#
proc MakeGeneList {w} {
    global spec

    frame $w
    listbox $w.l -relief raised -yscrollcommand "$w.s set"
    scrollbar $w.s -command "$w.l yview"
    set size 0
    foreach n [lsort [array names spec]] {
	$w.l insert end $n
        if { [string length $n] > $size } {
	    set size [string length $n]
	}
    }
    $w.l configure -width $size -height 10
    pack $w.l -side left
    pack $w.s -side left -fill y
    $w.l select set 0
}
    

#
# Get (if no value supplied) or Set (if a value supplied) the value in the 
# "Gene widget" (stored in chrom) 
#
proc GeneWidgetValue args {
    global chrom spec

    if { $args == "" } {
    	set gene ""
    	for {set i 0} {$i < [array size chrom]} {incr i} {
    	    set gene "$gene$chrom($i)"
    	}
    	return $gene
    } else {
	for {set i 0} {$i < [array size chrom]} {incr i} {
	    set chrom($i) [string index $spec($args) $i]
	}
    }
}

#
# Throw up a dialog the user can use to add, delete, view and edit genes
#
proc ModifyGenes {} {
    global spec ng_result spec1name spec2name

    MyDialogPrep ng "Add To List" "Read From List" "Delete From List"

    wm title .ng "Define new genes"

    MakeGeneList .ng.top.list
    MakeGeneWidget .ng.top.gene
    label .ng.top.gene.namelbl -text "New name:"
    entry .ng.top.gene.name -relief sunken
    bind .ng.top.gene.name <Return> {}
    pack .ng.top.gene.namelbl .ng.top.gene.name
    
    pack .ng.top.list .ng.top.gene -side left

    set oldFocus [focus]
    while { $ng_result == 0 } {
	grab set .ng
	focus .ng
	tkwait variable ng_result
	if { $ng_result==2 } { 
	    set newName [.ng.top.gene.name get]
	    if { $newName != "" && [lsearch [array names spec] $newName]==-1} {
		set spec($newName) [GeneWidgetValue]
		.ng.top.list.l insert \
		    [lsearch [lsort [array names spec]] $newName] $newName
		set w [lindex [.ng.top.list.l configure -width] 4]
		set h [lindex [.ng.top.list.l configure -height] 4]
		
		if { [string length $newName] > $w } {
		    .ng.top.list.l configure \
			-width "[string length $newName]"
		}
	    }
	    set ng_result 0
	} elseif { $ng_result==3 } {
	    set i [.ng.top.list.l curselection]
	    GeneWidgetValue [.ng.top.list.l get $i]
	    .ng.top.gene.name delete 0 end
	    .ng.top.gene.name insert end [.ng.top.list.l get $i]
	    .ng.top.list.l select clear 0 end
	    .ng.top.list.l select set end
	    
	    set ng_result 0
	} elseif { $ng_result==4 } {
	    set i [.ng.top.list.l curselection]
	    if { [.ng.top.list.l size]==1 } {
		tk_dialog .dlg "" \
		    "You must have at least one gene definition." warning 0 OK
	    } else {
	    	set i [.ng.top.list.l curselection]
	    	set name [.ng.top.list.l get $i]
	    	unset spec($name)
		.ng.top.list.l delete $i $i
		if { [.ng.top.list.l size] == $i } {
		    incr i -1
		}
		if { $name == $spec1name } {
		    set spec1name [.ng.top.list.l get 0]
		}
		if { $name == $spec2name } {
		    set spec2name [.ng.top.list.l get 0]
		}
		.ng.top.list.l select clear 0 end
		.ng.top.list.l select set $i
	    }
	    set ng_result 0
	}
    }
    destroy .ng
    focus $oldFocus
    unset ng_result    
    RebuildSpecMenus
}


#
# Toggle the choice between a Duel or a Test
#
proc DuelOrTest {} {
    global kind

    if { $kind == "duel" } {
	set kind "test"
	pack forget .status.spec2
	.mbar.test configure -state disabled
	.mbar.duel configure -state normal
    } else {
	set kind "duel"
	pack .status.spec2 -in .status.counts -side top -anchor nw
	.mbar.test configure -state normal
	.mbar.duel configure -state disabled
    }
}

#
# Read in a list of gene specifications from a file (whose name will 
# be asked for).  Duplicate names will have a suffix appended to them.
#
proc ReadGenes args {
    global spec 

    if { $args == "" } {
    	set fname [EntryDialog rg "Specify filename" "Read genes from:" \
	    "" 30 "%s"]
    } else {
        set fname $args
    }
    if { $fname != "" } {
	if { [file isfile $fname] == 1 } {
	    set fid [open $fname r]
	    set lineNbr 1;
	    while { [eof $fid]==0 } {
		set name [gets $fid]
	 	if { [eof $fid] == 1 } {
		    if { $name != "" } {
		    	tk_dialog .dlg "" "Premature end of file reached" \
			    warning 0 OK
		    }
		} else {
		    set gstring [gets $fid]
		    incr lineNbr 2
		    if { [scan $gstring "%50\[0123\]%s" data extra] == 1 &&
			[string length $data] == 50 } {
		    	    set gname $name
		    	    set i 2
		    	    while { [lsearch [array names spec] $gname] != -1 } {
				set gname "$name<$i>"
				incr i
		    	    }
		        set spec($gname) $gstring
		    } else {
			puts "Improper data format on line $lineNbr"
		    }
		}
	    }
	    close $fid
	    RebuildSpecMenus
	} else {
	    tk_dialog .dlg "" "Cannot read from '$fname'" warning 0 OK
	}
    }
}


#
# Write out the current gene specifications to a file (whose name will 
# be asked for).
#
proc WriteGenes {} {
    global spec

    set fname [EntryDialog rg "Specify filename" "Write genes to:" \
	"" 30 "%s"]
    if { $fname != "" } {
	if { [file isfile $fname]==0 || [file writable $fname] } {
	    set fid [open $fname w]
	    foreach s [array names spec] {
		puts $fid $s
		puts $fid $spec($s)
	    }
	    close $fid
	} else {
	    tk_dialog .dlg "" "Cannot read from '$fname'" warning 0 OK
	}
    }
}

#
# Rebuild the menus used to pick species for the test/duel.  This has to 
# be done whenever the list of species is altered.
#
proc RebuildSpecMenus {} {
    global spec

    destroy .status.spec1name.menu
    destroy .status.spec2name.menu
    menu .status.spec1name.menu
    menu .status.spec2name.menu
    foreach s [lsort [array names spec]] {
    	.status.spec1name.menu add radiobutton -label $s -variable spec1name \
	    -value $s
        .status.spec2name.menu add radiobutton -label $s -variable spec2name \
	    -value $s
    }
}

# ------------------------------------------------
# ------------- execution procedures -------------

#
# Start a simulation run
#
proc RunSimulation {} {
     global status kind

     set status run
    .mbar.exec.menu entryconfigure Run -state disabled
    .mbar.exec.menu entryconfigure Step -state disabled
    .status.step configure -state disabled
    .mbar.exec.menu entryconfigure Stop -state normal
    .mbar.exec.menu entryconfigure Reset -state disabled
    .mbar.gene configure -state disabled
    .mbar.display configure -state disabled
    if { $kind == "test" } {
    	.mbar.duel configure -state disabled
    } else {
    	.mbar.test configure -state disabled
    }
    .status.spec1name configure -state disabled
    .status.spec2name configure -state disabled
    RunSim run
}

#
# Run the simulation for a single step
#
proc StepSimulation {} {
    global status kind

    if { $status != "step" } {
        set status step
        .mbar.exec.menu entryconfigure Run -state disabled
        .mbar.exec.menu entryconfigure Step -state disabled
	.status.step configure -state disabled
        .mbar.exec.menu entryconfigure Stop -state normal
	.mbar.exec.menu entryconfigure Reset -state disabled
    	.mbar.gene configure -state disabled
        .mbar.display configure -state disabled
    	if { $kind == "test" } {
    	    .mbar.duel configure -state disabled
    	} else {
    	    .mbar.test configure -state disabled
        }
	.status.spec1name configure -state disabled
	.status.spec2name configure -state disabled
    }
    RunSim step
}

#
# Stop the simulation
#
proc StopSimulation {} {
    global status

    SetStatus stop
    set status stop
    .mbar.exec.menu entryconfigure Run -state normal
    .mbar.exec.menu entryconfigure Step -state normal
    .status.step configure -state normal
    .mbar.exec.menu entryconfigure Stop -state disabled
    .mbar.exec.menu entryconfigure Reset -state normal
}

#
# Re-initialize the simulation
#
proc ResetSimulation {} {
    global status kind

    SetStatus reset
    set status reset
    .mbar.exec.menu entryconfigure Run -state normal
    .mbar.exec.menu entryconfigure Step -state normal
    .status.step configure -state disabled
    .mbar.exec.menu entryconfigure Stop -state disabled
    .mbar.exec.menu entryconfigure Reset -state disabled
    .mbar.gene configure -state normal
    .mbar.display configure -state normal
    if { $kind == "test" } {
    	.mbar.duel configure -state normal
    } else {
    	.mbar.test configure -state normal
    }
    .status.spec1name configure -state normal
    .status.spec2name configure -state normal
    ClearArena
}

#
# Exit the simulator
#
proc QuitSimulator {} {
    SetStatus reset
    set status reset
    exit
}


# define the main window -- menus, status and arena
frame .mbar -relief raised -bd 2
frame .status -relief raised -bd 2
frame .arena -relief raised -bd 2
pack append . .mbar {top fillx} .status {top fillx} .arena {top frame center}

# define the menu bar
menubutton .mbar.exec -text Exec -menu .mbar.exec.menu
menubutton .mbar.gene -text Genes -menu .mbar.gene.menu
menubutton .mbar.display -text Display -menu .mbar.display.menu
pack .mbar.exec .mbar.gene .mbar.display -side left

button .mbar.test -text Test -command DuelOrTest
button .mbar.duel -text Duel -command DuelOrTest -state disabled
pack .mbar.test .mbar.duel -side right

menu .mbar.exec.menu
.mbar.exec.menu add command -label "Run" -command "RunSimulation"
.mbar.exec.menu add command -label Step -command "StepSimulation"
.mbar.exec.menu add command -label Stop -command "StopSimulation" \
    -state disabled
.mbar.exec.menu add separator
.mbar.exec.menu add command -label Reset -command "ResetSimulation" \
    -state disabled
.mbar.exec.menu add separator
.mbar.exec.menu add command -label Quit -command QuitSimulator

menu .mbar.gene.menu
.mbar.gene.menu add command -label "Modify List..." -command ModifyGenes
.mbar.gene.menu add command -label "Read Genes..." -command ReadGenes
.mbar.gene.menu add command -label "Write Genes..." -command WriteGenes

menu .mbar.display.menu
.mbar.display.menu add checkbutton -label "Display each round" \
    -variable showRounds 
.mbar.display.menu add separator
.mbar.display.menu add radiobutton -label "Small" -variable size -value 16 \
    -command "ClearArena"
.mbar.display.menu add radiobutton -label "Medium" -variable size -value 32 \
    -command "ClearArena"
.mbar.display.menu add radiobutton -label "Large" -variable size -value 48 \
    -command "ClearArena"

# define the status line
label .status.clockCountlabel -text "Rounds: "
label .status.clockCountvalue -textvariable display_clockCount -relief ridge -width 6
pack .status.clockCountlabel .status.clockCountvalue -side left -anchor nw

frame .status.counts -bd 0
frame .status.spec1 -bd 0
label .status.spec1pic -bitmap pac16_000 -fg $color(spec1)
label .status.spec1value -textvariable spec1Cnt -relief ridge \
	-width 3
menubutton .status.spec1name -textvariable spec1name  -fg $color(spec1) \
    -menu .status.spec1name.menu
pack .status.spec1pic .status.spec1value .status.spec1name \
	-in .status.spec1 -side left

frame .status.spec2 -bd 0
label .status.spec2pic -bitmap pac16_100 -fg $color(spec2)
label .status.spec2value -textvariable spec2Cnt -relief ridge \
	-width 3
menubutton .status.spec2name -textvariable spec2name  -fg $color(spec2) \
    -menu .status.spec2name.menu
pack .status.spec2pic .status.spec2value .status.spec2name \
	-in .status.spec2 -side left

pack .status.spec1 .status.spec2 -in .status.counts -side top -anchor nw
pack .status.counts -side left -padx 3m -anchor nw

menu .status.spec1name.menu
menu .status.spec2name.menu
RebuildSpecMenus

button .status.step -text Step -command StepSimulation -state disabled
pack .status.step -side right -anchor ne

InitArena
update
ReadGenes class.data
