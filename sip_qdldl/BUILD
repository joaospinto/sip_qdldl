cc_library(
	name = "sparse",
	srcs = ["sparse.cpp",],
	hdrs = ["sparse.hpp",],
  deps = [],
)

cc_library(
	name = "sip_qdldl",
	srcs = ["sip_qdldl.cpp", "sparse.cpp",],
	hdrs = ["sip_qdldl.hpp", "sparse.hpp",],
  deps = [
    "@qdldl//:qdldl",
    ":sparse",
  ],
  visibility = ["//visibility:public",],
)
