This README has steps to provide access to NVIDIA NeMo Multimodal Release 23.09 (EA) container. The container is pushed to team `ea-mm-sd-alpha` under `ea-bignlp` project. The container can be pulled as follows 
```
docker pull nvcr.io/ea-bignlp/ea-mm-sd-alpha/bignlp-mm-sd:23.09-py3
```

# Approve Early Access request
For users within NVIDIA, skip this section and go to `Give access to NGC container` directly. For users outside NVIDIA, they need to request access to the container on `https://developer.nvidia.com/nemo-framework-multimodal-early-access` and specify that they want access to the `MLPerf` container. 

Once filled, people with access to DevZone should log into DevZone and be connected to NGVPN02 to access the program `https://developer.nvidia.com/nv/admin/programs?locale=en`. There, requests can be approved. Once approved, add the email to NGC manually using steps in the next section.

## DevZone access
If you do not have access to DevZone to approve new requests, reach out to Arham Mehta who is a PM for Nemo MultiModal and create a jira like this one `https://jirasw.nvidia.com/browse/MLPTRN-2041` and assign to Jeff Bowker. 

# Give access to NGC container
To give access to a new user, add their email to `https://org.ngc.nvidia.com/teams`> Invite New User and give them `Private Registry Read` permissions. After this, the user will get an invitation on their email which they need to accept before pulling the container.

## NGC access
Reach out to Shriya/Michal/Matt/Burc to get added to NGC `ea-mm-sd-alpha/ea-bignlp`
