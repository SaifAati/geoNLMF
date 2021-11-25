message("----------cmake src ----------")

find_package(Threads REQUIRED)


add_library(mainNLMF OBJECT geoNLMF.cpp)
add_library(geoNLMFLib OBJECT geoNLMFLib.cpp)


add_library(geoNLMF SHARED)
target_link_libraries(geoNLMF mainNLMF geoNLMFLib)
message("----------end cmake src ----------")