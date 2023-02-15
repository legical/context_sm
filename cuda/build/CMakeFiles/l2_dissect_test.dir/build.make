# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bric/Workspace/context_sm/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bric/Workspace/context_sm/cuda/build

# Include any dependencies generated for this target.
include CMakeFiles/l2_dissect_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/l2_dissect_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/l2_dissect_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/l2_dissect_test.dir/flags.make

CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o: CMakeFiles/l2_dissect_test.dir/flags.make
CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o: ../src/dissect_page/dissect_page.cu
CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o: CMakeFiles/l2_dissect_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bric/Workspace/context_sm/cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o -MF CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o.d -x cu -c /home/bric/Workspace/context_sm/cuda/src/dissect_page/dissect_page.cu -o CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o

CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target l2_dissect_test
l2_dissect_test_OBJECTS = \
"CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o"

# External object files for target l2_dissect_test
l2_dissect_test_EXTERNAL_OBJECTS =

l2_dissect_test: CMakeFiles/l2_dissect_test.dir/src/dissect_page/dissect_page.cu.o
l2_dissect_test: CMakeFiles/l2_dissect_test.dir/build.make
l2_dissect_test: CMakeFiles/l2_dissect_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bric/Workspace/context_sm/cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable l2_dissect_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/l2_dissect_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/l2_dissect_test.dir/build: l2_dissect_test
.PHONY : CMakeFiles/l2_dissect_test.dir/build

CMakeFiles/l2_dissect_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/l2_dissect_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/l2_dissect_test.dir/clean

CMakeFiles/l2_dissect_test.dir/depend:
	cd /home/bric/Workspace/context_sm/cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bric/Workspace/context_sm/cuda /home/bric/Workspace/context_sm/cuda /home/bric/Workspace/context_sm/cuda/build /home/bric/Workspace/context_sm/cuda/build /home/bric/Workspace/context_sm/cuda/build/CMakeFiles/l2_dissect_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/l2_dissect_test.dir/depend

