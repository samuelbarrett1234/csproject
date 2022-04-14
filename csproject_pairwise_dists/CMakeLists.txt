cmake_minimum_required (VERSION 3.8)

project ("csproject_pairwise_dists")

find_package (SQLite3)

add_executable(csproject_pairwise_dists csproject_pairwise_dists.cpp)
target_compile_features(csproject_pairwise_dists PRIVATE cxx_std_17)

if (SQLITE3_FOUND)
  message("Found SQLite headers at " ${SQLite3_INCLUDE_DIRS})
  include_directories(${SQLite3_INCLUDE_DIRS})
  target_link_libraries (csproject_pairwise_dists ${SQLite3_LIBRARIES})
endif (SQLITE3_FOUND)
