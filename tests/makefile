# MOAB_DIR points to top-level install dir, below which MOAB's lib/ and include/ are located
include ../makefile.config

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  MOAB_CXXFLAGS += -fopenmp -g -O3
  #MOAB_CXXFLAGS += -save-temps -v
  MOAB_LIBS_LINK += -fopenmp
else # Darwin
  MOAB_CXXFLAGS += -g -O3 -std=c++17 -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/21.1.3/include
  MOAB_LIBS_LINK += -L/opt/homebrew/Cellar/libomp/21.1.3/lib -lomp -framework Accelerate
endif

MOAB_CXXFLAGS += -I../include # add include directory for the tool

SRC_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp, %.o, $(SRC_FILES))

%.o: %.cpp
	@echo "  [CXX]  $< ..."
	${VERBOSE}${MOAB_CXX} ${CXXFLAGS} ${MOAB_CXXFLAGS} ${MOAB_CPPFLAGS} ${MOAB_INCLUDES} -DMESH_DIR=\"${MESH_DIR}\" -c $< -o $@

default: all

EXAMPLES = test_compare_rgl_kdtree_locator test_kdtree_locator
ALLEXAMPLES = ${EXAMPLES}

all: $(ALLEXAMPLES)

# Compile RegularGridLocator directly — avoids pulling in mbda.o (which owns
# main + easylogging globals) and other unrelated objects from ../.objs/.
RegularGridLocator.o: ../src/RegularGridLocator.cpp
	@echo "  [CXX]  $< ..."
	${VERBOSE}${MOAB_CXX} ${CXXFLAGS} ${MOAB_CXXFLAGS} ${MOAB_CPPFLAGS} ${MOAB_INCLUDES} -DMESH_DIR=\"${MESH_DIR}\" -c $< -o $@

test_compare_rgl_kdtree_locator: test_compare_rgl_kdtree_locator.o RegularGridLocator.o
	@echo "[CXXLD]  $@ ..."
	${VERBOSE}${MOAB_CXX} -o $@ $^ ${MOAB_LIBS_LINK}

# KD-tree test is standalone (no easylogging, no RegularGridLocator)
test_kdtree_locator: test_kdtree_locator.o
	@echo "[CXXLD]  $@ ..."
	${VERBOSE}${MOAB_CXX} -o $@ $^ ${MOAB_LIBS_LINK}

run: all
	@for t in $(ALLEXAMPLES); do echo "=== Running $$t ===" && ./$$t && echo "" || exit 1; done

clean: clobber
	rm -rf ${ALLEXAMPLES}
	rm -rf $(OBJ_FILES) RegularGridLocator.o
