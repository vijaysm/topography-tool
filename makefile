# MOAB_DIR points to top-level install dir, below which MOAB's lib/ and include/ are located
include makefile.config

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  MOAB_CXXFLAGS += -fopenmp -g -O2
  MOAB_LIBS_LINK += -fopenmp
else
  MOAB_CXXFLAGS += -g -std=c++17 -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/21.1.0/include
  MOAB_LIBS_LINK = -L/opt/homebrew/Cellar/libomp/21.1.0/lib -lomp
endif

MOAB_CXXFLAGS += -Iinclude # add include directory for the tool

SRC_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(patsubst src/%.cpp, .objs/%.o, $(SRC_FILES))
LINK_FILES := $(patsubst src/%.cpp, %.o, $(SRC_FILES))

.objs/%.o: src/%.cpp
	@echo "  [CXX]  $< ..."
	${VERBOSE}${MOAB_CXX} ${CXXFLAGS} ${MOAB_CXXFLAGS} ${MOAB_CPPFLAGS} ${MOAB_INCLUDES} -DMESH_DIR=\"${MESH_DIR}\" -c $< -o $@

default: all

EXAMPLES =  TOPORemapper
ALLEXAMPLES = ${EXAMPLES}

all: $(ALLEXAMPLES)

#TOPORemapper: src/TOPORemapper.o src/ParallelPointCloudReader.o src/ParallelPointCloudDistributor.o src/ScalarRemapper.o
TOPORemapper: $(OBJ_FILES)
	@echo "[CXXLD]  $@ ..."
	${VERBOSE}${MOAB_CXX} -o $@ ${OBJ_FILES} ${MOAB_LIBS_LINK}

run: all $(addprefix run-,$(ALLEXAMPLES))

clean: clobber
	rm -rf ${ALLEXAMPLES}
	rm -rf $(OBJ_FILES)

