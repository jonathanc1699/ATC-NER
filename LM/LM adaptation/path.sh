export KALDI_ROOT=/home/jonathan/Documents/Intern/kaldi
export LD_LIBRARY_PATH=/cm/shared/apps/cuda10.0/toolkit/10.0.130/lib64:/home4/md510/gcc-5.2.0/lib64:/cm/shared/apps/cuda10.0/toolkit/10.0.130/extras/CUPTI/lib64:/cm/local/apps/cuda/libs/current/lib64:/cm/shared/apps/cuda10.0/toolkit/10.0.130/targets/x86_64-linux/lib:/cm/shared/apps/slurm/14.03.0/lib64/slurm:/cm/shared/apps/slurm/14.03.0/lib64:/cm/shared/apps/gcc/4.8.4/lib:/cm/shared/apps/gcc/4.8.4/lib64:/cm/shared/apps/boost-1.60.0/lib:/cm/shared/apps/mkl_intel/mkl/lib/intel64:/cm/shared/apps/cudnn-5.1/cuda/lib64:/cm/shared/apps/cuda75/toolkit/7.5.18/lib64:/cm/shared/apps/mpi/lib:$LD_LIBRARY_PATH

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/kaldi_lm:/home/hhx502/w2020/kaldi/tools/srilm/lm/bin/i686-m64:$PWD:$PATH:/home4/hhx502/w2018/rar
export PATH=$KALDI_ROOT/tools/srilm/bin/i686-m64:$PATH
export PATH=$KALDI_ROOT/tools/sph2pipe_v2.5:/home/opt/tools/NIST/sctk-2.4.8/bin:/home2/hhx502/cntk-oct-04-2016/build/release/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
