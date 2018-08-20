if [ $1 ]; then
	arch=$1
else
	arch=OpenMP
fi
make arch=$arch clean
make arch=$arch
