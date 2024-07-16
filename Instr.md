## To run 

1. Install the requirements

  ```bash
  pip install -r requirements.txt
  ```
2. Setup redis on docker:

i) Install the docker image of redis

ii) ```
   docker run -d -p 6379:6379 --name rdb redis
     ```
     
iii) ```
   docker start rdb
     ```
