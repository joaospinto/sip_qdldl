load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
	name = "sparse",
	srcs = ["sparse.cpp",],
	hdrs = ["sparse.hpp",],
  deps = [],
)

cc_library(
	name = "sip_qdldl",
	srcs = ["sip_qdldl.cpp",],
	hdrs = ["sip_qdldl.hpp",],
  deps = [
    "@qdldl//:qdldl",
    ":sparse",
  ],
  visibility = ["//visibility:public",],
)
