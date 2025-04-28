SkyPilot
========

.. attention:: 
    To be updated for Qwen3.

What is SkyPilot
----------------

SkyPilot is a framework for running LLMs, AI, and batch jobs on any
cloud, offering maximum cost savings, the highest GPU availability, and
managed execution. Its features include:

-  Get the best GPU availability by utilizing multiple resources pools
   across multiple regions and clouds.
-  Pay absolute minimum — SkyPilot picks the cheapest resources across
   regions and clouds. No managed solution markups.
-  Scale up to multiple replicas across different locations and
   accelerators, all served with a single endpoint
-  Everything stays in your cloud account (your VMs & buckets)
-  Completely private - no one else sees your chat history

Install SkyPilot
----------------

We advise you to follow the
`instruction <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html>`__
to install SkyPilot. Here we provide a simple example of using ``pip``
for the installation as shown below.

.. code:: bash

   # You can use any of the following clouds that you have access to:
   # aws, gcp, azure, oci, lamabda, runpod, fluidstack, paperspace,
   # cudo, ibm, scp, vsphere, kubernetes
   pip install "skypilot-nightly[aws,gcp]"

After that, you need to verify cloud access with a command like:

.. code:: bash

   sky check

For more information, check the `official document <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html>`__ and see if you have
set up your cloud accounts correctly.

Alternatively, you can also use the official docker image with SkyPilot
master branch automatically cloned by running:

.. code:: bash

   # NOTE: '--platform linux/amd64' is needed for Apple Silicon Macs
   docker run --platform linux/amd64 \
     -td --rm --name sky \
     -v "$HOME/.sky:/root/.sky:rw" \
     -v "$HOME/.aws:/root/.aws:rw" \
     -v "$HOME/.config/gcloud:/root/.config/gcloud:rw" \
     berkeleyskypilot/skypilot-nightly

   docker exec -it sky /bin/bash

Running Qwen2.5-72B-Instruct with SkyPilot
------------------------------------------

1. Start serving Qwen2.5-72B-Instruct on a single instance with any
   available GPU in the list specified in
   `serve-72b.yaml <https://github.com/skypilot-org/skypilot/blob/master/llm/qwen/serve-72b.yaml>`__
   with a vLLM-powered OpenAI-compatible endpoint:
   
   .. code:: bash

      sky launch -c qwen serve-72b.yaml

   **Before launching, make sure you have changed Qwen/Qwen2-72B-Instruct to Qwen/Qwen2.5-72B-Instruct in the YAML file.**

2. Send a request to the endpoint for completion:

   .. code:: bash

      IP=$(sky status --ip qwen)

      curl -L http://$IP:8000/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "prompt": "My favorite food is",
            "max_tokens": 512
      }' | jq -r '.choices[0].text'

3. Send a request for chat completion:

   .. code:: bash

      curl -L http://$IP:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [
            {
               "role": "system",
               "content": "You are Qwen, created by Alibaba Cloud. You are a helpful and honest chat expert."
            },
            {
               "role": "user",
               "content": "What is the best food?"
            }
            ],
            "max_tokens": 512
      }' | jq -r '.choices[0].message.content'

Scale up the service with SkyPilot Serve
----------------------------------------

1. With `SkyPilot
   Serve <https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html>`__,
   a serving library built on top of SkyPilot, scaling up the Qwen
   service is as simple as running:

   .. code:: bash

      sky serve up -n qwen ./serve-72b.yaml

   **Before launching, make sure you have changed Qwen/Qwen2-72B-Instruct to Qwen/Qwen2.5-72B-Instruct in the YAML file.**

   This will start the service with multiple replicas on the cheapest
   available locations and accelerators. SkyServe will automatically manage
   the replicas, monitor their health, autoscale based on load, and restart
   them when needed.

   A single endpoint will be returned and any request sent to the endpoint
   will be routed to the ready replicas.

2. To check the status of the service, run:

   .. code:: bash

      sky serve status qwen

   After a while, you will see the following output:

   ::

      Services
      NAME        VERSION  UPTIME  STATUS        REPLICAS  ENDPOINT            
      Qwen  1        -       READY         2/2       3.85.107.228:30002  

      Service Replicas
      SERVICE_NAME  ID  VERSION  IP  LAUNCHED    RESOURCES                   STATUS REGION  
      Qwen          1   1        -   2 mins ago  1x Azure({'A100-80GB': 8}) READY  eastus  
      Qwen          2   1        -   2 mins ago  1x GCP({'L4': 8})          READY  us-east4-a 

   As shown, the service is now backed by 2 replicas, one on Azure and one
   on GCP, and the accelerator type is chosen to be **the cheapest
   available one** on the clouds. That said, it maximizes the availability
   of the service while minimizing the cost.

3. To access the model, we use a ``curl -L`` command (``-L`` to follow
   redirect) to send the request to the endpoint:

   .. code:: bash

      ENDPOINT=$(sky serve status --endpoint qwen)

      curl -L http://$ENDPOINT/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [
            {
               "role": "system",
               "content": "You are Qwen, created by Alibaba Cloud. You are a helpful and honest code assistant expert in Python."
            },
            {
               "role": "user",
               "content": "Show me the python code for quick sorting a list of integers."
            }
            ],
            "max_tokens": 512
      }' | jq -r '.choices[0].message.content'

Accessing Qwen2.5 with Chat GUI
---------------------------------------------

It is also possible to access the Qwen2.5 service with GUI by connecting a
`FastChat GUI server <https://github.com/lm-sys/FastChat>`__ to the endpoint launched
above (see `gui.yaml <https://github.com/skypilot-org/skypilot/blob/master/llm/qwen/gui.yaml>`__).

1. Start the Chat Web UI:

   .. code:: bash

      sky launch -c qwen-gui ./gui.yaml --env ENDPOINT=$(sky serve status --endpoint qwen)

   **Before launching, make sure you have changed Qwen/Qwen1.5-72B-Chat to Qwen/Qwen2.5-72B-Instruct in the YAML file.**

2. Then, we can access the GUI at the returned gradio link:

   ::

      | INFO | stdout | Running on public URL: https://6141e84201ce0bb4ed.gradio.live

   Note that you may get better results by using a different temperature and top_p value.

Summary
-------

With SkyPilot, it is easy for you to deploy Qwen2.5 on any cloud. We
advise you to read the official doc for more usages and updates.
Check `this <https://skypilot.readthedocs.io/>`__ out!
