docker build -f ./Dockerfile -t tf .

docker run -v $(pwd):/normative -it tf bash ./commands_list.sh