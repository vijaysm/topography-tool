# MOAB_DIR points to top-level install dir, below which MOAB's lib/ and include/ are located
include makefile.config

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  MOAB_CXXFLAGS += -fopenmp -g -O3
  #MOAB_CXXFLAGS += -save-temps -v
  MOAB_LIBS_LINK += -fopenmp
else # Darwin
  MOAB_CXXFLAGS += -g -O3 -std=c++17 -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/21.1.3/include
  MOAB_LIBS_LINK += -L/opt/homebrew/Cellar/libomp/21.1.3/lib -lomp -framework Accelerate
endif

MOAB_CXXFLAGS += -Iinclude # add include directory for the tool

SRC_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(patsubst src/%.cpp, .objs/%.o, $(SRC_FILES))
LINK_FILES := $(patsubst src/%.cpp, %.o, $(SRC_FILES))

.objs/%.o: src/%.cpp
	@echo "  [CXX]  $< ..."
	${VERBOSE}${MOAB_CXX} ${CXXFLAGS} ${MOAB_CXXFLAGS} ${MOAB_CPPFLAGS} ${MOAB_INCLUDES} -DMESH_DIR=\"${MESH_DIR}\" -c $< -o $@

default: .dummy all

EXAMPLES =  mbda
ALLEXAMPLES = ${EXAMPLES}

all: $(ALLEXAMPLES)

#mbda: src/mbda.o src/ParallelPointCloudReader.o src/ParallelPointCloudDistributor.o src/ScalarRemapper.o
mbda: $(OBJ_FILES)
	@echo "[CXXLD]  $@ ..."
	${VERBOSE}${MOAB_CXX} -o $@ ${OBJ_FILES} ${MOAB_LIBS_LINK}

run: all $(addprefix run-,$(ALLEXAMPLES))

test-regression:
	bash tests/regression_checks.sh

format:
	find . -iname '*.hpp' -o -iname '*.cpp' | xargs clang-format -i --style=LLVM

.dummy:
	@mkdir -p .objs

clean: clobber
	rm -rf ${ALLEXAMPLES}
	rm -rf $(OBJ_FILES)
