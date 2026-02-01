#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#define ACCELGEN_DEBUG

#ifdef ACCELGEN_DEBUG
#define ECHO(content, split) llvm::errs() << content << split;
#define ECHO_LIST(list, split)                                                 \
  llvm::interleave(list, llvm::errs(), split);                                 \
  llvm::errs() << "\n";
#else
#define ECHO(content, split) ;
#define ECHO_LIST(list, split) ;

#endif

#endif
