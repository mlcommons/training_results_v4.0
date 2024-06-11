#!/bin/bash

#####################################################
#### HELPER FUNCTIONS
#####################################################
function test_server_access {
    if ! curl -sL --connect-timeout 3 --fail http://cuda-repo/ --output /dev/null --silent; then
        echo "WARNING: not able to access http://cuda-repo/. Probably this script won't work well."
        echo "    This script can run on Nvidia Intranet which has access to http://cuda-repo."
        echo "    If you have access to the repo, please check the network settings to resolve the ip address for cuda-repo."
    fi
}
test_server_access

function find_matching_version_in_autoindex {
    # args
    local INDEXURL=$1
    local KEY="^$2/$"
    find_matching_url_in_autoindex "${INDEXURL}" "${KEY}"
}

function find_matching_url_in_autoindex {
    # args
    local INDEXURL=$1
    local KEY=$2
    if [ -z "${INDEXURL}" -o -z "${KEY}" ]; then
        echo "Missing arguments for find_matching_url_in_autoindex"
        return 1
    fi
    # tokenize autoindex return, grab href
    local INDEX=$(curl -ucuda_ro:Nvidia3d! -sL --connect-timeout 30 --retry 10 --retry-delay 0 "${INDEXURL}?C=M;O=A" | sed -n 's/.*href=["'\'']\([^"'\'']*\).*/\1/p')
    local MATCH=$(echo "${INDEX}" | grep "${KEY}")
    if [ $(echo "${MATCH}" | wc -l) -gt 1 ]; then
        echo "========================" >&2
        echo "WARNING: Multiple matches for '${KEY}' in index ${INDEXURL}" >&2
        echo "========================" >&2
        echo "${MATCH}" >&2
        echo "========================" >&2
        MATCH=$(echo "${MATCH}" | tail -n 1)
    fi
    local HREF=$(echo "${MATCH}" | sed 's/[<>=" ]\+/,/g' | cut -d , -f 4)
    if [ -z "${HREF}" ]; then
        echo "========================" >&2
        echo "ERROR: key '${KEY}' not found in index ${INDEXURL/https:\/\/*:*@/https:\/\/}" >&2
        echo "========================" >&2
        echo "${INDEX}" | grep "a href" | grep -v "Last modified" >&2
        echo "========================" >&2
        return 1
    fi
    echo "${INDEXURL}/${HREF}"
}

#####################################################
#### find the latest dvs package in a url
#####################################################
function find_newest_dvs_package {
    # args
    local INDEXURL=$1
    local COMPONENT=$2
    if [ -z "${INDEXURL}" -o -z "${COMPONENT}" ]; then
        echo "Missing arguments for find_newest_dvs_package"
        return 1
    fi
    # search for packages with a pattern and find the newest
    local CHANGELIST=$(curl -sL "${INDEXURL}" | cut -d \" -f 2 | grep ${COMPONENT}.tgz | sort -V | tail -n 1 | cut -d "_" -f 2 | cut -d "." -f 1)
    if [ -z "${CHANGELIST}" ]; then
        echo "========================" >&2
        echo "ERROR: unable to find the latest package of type '${COMPONENT}'.tgz in index '${INDEXURL}'" >&2
        echo "========================" >&2
        return 1
    fi
    echo "${CHANGELIST}"
}

#####################################################
#### find the package download url
#####################################################
function find_dvs_package_download_url {
    #args
    local DVS_PACKAGE_NAME=$1
    local DVS_CHANGELIST=$2
    local DVS_LOCATION=${3:ausdvs}
    if [ -z "${DVS_PACKAGE_NAME}" -o -z "${DVS_CHANGELIST}" ]; then
        echo "Missing arguments for find_dvs_package_download_url"
        return 1
    fi
    local DVS_QUERY_URL="http://${DVS_LOCATION}.nvidia.com/Query/User?which_request=package_url_request&which_package=${DVS_PACKAGE_NAME}&which_changelist=${DVS_CHANGELIST}"
    local DVS_PACKAGE_URL=$(curl -sL "${DVS_QUERY_URL}" | tail -1)
    if [[ "${DVS_PACKAGE_URL}" == "DOESNT_EXIST" ]]; then
        echo "==========================" >&2
        echo "ERROR: DVS_QUERY_URL ${DVS_QUERY_URL} does not return a package download url" >&2
        echo "==========================" >&2
        return 1
    fi
    if ! url_exists "${DVS_PACKAGE_URL}"; then
        echo "==========================" >&2
        echo "ERROR: unable to find the package download url for ${DVS_PACKAGE_NAME} at changelist ${DVS_CHANGELIST}" >&2
        echo "==========================" >&2
        return 1
    fi
    echo "${DVS_PACKAGE_URL}"
}

#####################################################
#### find if a url is available
#####################################################
function url_exists() {
    curl -ucuda_ro:Nvidia3d! --connect-timeout 30 --retry 10 --retry-delay 0 --output /dev/null --silent --head --fail "$1"
    return $ret
}

function find_if_url_exists {
    # args
    local URLTOCHECK=$1
    if curl -ucuda_ro:Nvidia3d! --connect-timeout 30 --retry 10 --retry-delay 0 --output /dev/null --silent --head --fail "$URLTOCHECK"; then
        echo 0
    else
        echo 1
    fi
}

#####################################################
#### find the latest release build available
#####################################################
function find_latest_release_version {
    # args
    local COMPONENT_NAME=$1
    local COMPONENT_BRANCH=$2
    local COMPONENT_ARCH=$3
    local COMPONENT_DISTRO_ARG=$(if [ -n $4 ]; then echo --distro=$4; fi)
    local SVC_USER=cuda_ro
    local SVC_PASS=Nvidia3d!
    local _SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    if [[ "${COMPONENT_BRANCH}" != "general" ]]; then COMPONENT_BRANCH=r${COMPONENT_BRANCH}; fi
    if [[ "${COMPONENT_NAME}" == *'libcublas'* ]]; then
        ! url_exists "${ARTIFACTORY}/${COMPONENT_BRANCH}/${CUBLAS_VERSION}" || COMPONENT_BRANCH="${COMPONENT_BRANCH}_CUBLAS_GPGPU"
    fi
    local COMPONENT_VERSION=$(SVC_USER=${SVC_USER} SVC_PASS=${SVC_PASS} ${_SCRIPT_DIR}/queryBuilds.sh --packages --os=linux --extension=deb --arch=${COMPONENT_ARCH} --branch=${COMPONENT_BRANCH} --component=${COMPONENT_NAME} ${COMPONENT_DISTRO_ARG} --print_version --tot || echo 0)
    echo "${COMPONENT_VERSION}"
}

#####################################################
#### find all the latest release builds available
#####################################################
function find_all_latest_release_versions {
    declare -A COMPONENT_TO_ARTIFACTORY_MAP=(
        ["CUDA_DRIVER"]="nvidia_driver"
        ["CUDA"]="cuda_cudart"
        ["CUBLAS"]="libcublas"
        ["CUFFT"]="libcufft"
        ["CURAND"]="libcurand"
        ["CUSPARSE"]="libcusparse"
        ["CUSOLVER"]="libcusolver"
        ["NPP"]="libnpp"
        ["NVJPEG"]="libnvjpeg"
        ["NSIGHT_COMPUTE"]="nsight_compute"
    )
    local COMPONENTS_IN_CUDA_REPO=("CUDNN" "NCCL" "NSIGHT_SYSTEMS" "TRT")
    #declare -A COMPONENTS_IN_DVS_MAP=( ["CUBLAS"]="gpu_drv_cuda_a_Release_Linux_AMD64_GPGPU_CUDA_CUBLAS" )
    declare -A COMPONENTS_IN_DVS_MAP=()
    local OTHER_VERSIONS_NOT_UPDATED=()

    for _COMPONENT in "${OTHER_VERSIONS_NOT_UPDATED[@]}"; do
        _COMPONENT_VERSION=${_COMPONENT}_VERSION
        echo "${_COMPONENT}_VERSION=${!_COMPONENT_VERSION}"
    done
    # args
    ## Parse the VERSION string to get branch and patch details
    for _COMPONENT in "${!COMPONENT_TO_ARTIFACTORY_MAP[@]}"; do
        _COMPONENT_VERSION=${_COMPONENT}_VERSION
        if [[ -z "${!_COMPONENT_VERSION}" ]]; then
            continue #skip to next component
        fi

        IFS='.' read -r -a _COMPONENT_VERSION_ARR <<<"${!_COMPONENT_VERSION}"
        local _COMPONENT_VERSION_MAJOR=${_COMPONENT_VERSION_ARR[0]}
        if [[ ${_COMPONENT} != "CUDA_DRIVER" ]]; then
            local _COMPONENT_VERSION_MINOR=${_COMPONENT_VERSION_ARR[1]}
            if [[ ${_COMPONENT} != "CUDA" ]]; then
                local _COMPONENT_VERSION_BUILD=${_COMPONENT_VERSION_ARR[2]}
            fi
            local _COMPONENT_BRANCH="${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"
            local _COMPONENT_DISTRO=""
        else
            local _CUDA_DRIVER_VERSION_MAJOR=${_COMPONENT_VERSION_MAJOR}
            local _COMPONENT_BRANCH="${_COMPONENT_VERSION_MAJOR}_00"
            local _COMPONENT_DISTRO="${DISTRO_NAME,,}${DISTRO_VERSION_MAJOR}${DISTRO_VERSION_MINOR}"
        fi
        local _COMPONENT_ARCH="${ARCH_NAME}"
        local _COMPONENT_VERSION_LATEST="$(find_latest_release_version ${COMPONENT_TO_ARTIFACTORY_MAP[${_COMPONENT}]} ${_COMPONENT_BRANCH} ${_COMPONENT_ARCH} ${_COMPONENT_DISTRO})"
        echo "${_COMPONENT}_VERSION=${_COMPONENT_VERSION_LATEST}${_COMPONENT_VERSION_SUFFIX}"
    done

    for _COMPONENT in "${!COMPONENTS_IN_DVS_MAP[@]}"; do
        local DVS="dvs-binaries"
        local DVSTRANSFERURL="http://dvstransfer.nvidia.com/dvsshare"
        PACKAGE=${COMPONENTS_IN_DVS_MAP[${_COMPONENT}]}

        _COMPONENT_VERSION=${_COMPONENT}_VERSION
        if [[ -z "${!_COMPONENT_VERSION}" ]]; then
            continue #skip to next component
        fi
        IFS='.' read -r -a _COMPONENT_VERSION_ARR <<<"${!_COMPONENT_VERSION}"
        local _COMPONENT_VERSION_MAJOR=${_COMPONENT_VERSION_ARR[0]}
        local _COMPONENT_VERSION_MINOR=${_COMPONENT_VERSION_ARR[1]}
        local _COMPONENT_VERSION_BUILD=$(find_newest_dvs_package "${DVSTRANSFERURL}/${DVS}/${PACKAGE}" "${PACKAGE}")
        local _COMPONENT_VERSION_LATEST="${_COMPONENT_VERSION_MAJOR}.${_COMPONENT_VERSION_MINOR}.${_COMPONENT_VERSION_BUILD}.0"
        echo "${_COMPONENT}_VERSION=${_COMPONENT_VERSION_LATEST}"
    done

    for _COMPONENT in "${COMPONENTS_IN_CUDA_REPO[@]}"; do
        _COMPONENT_VERSION=${_COMPONENT}_VERSION
        if [ -z "${!_COMPONENT_VERSION}" ]; then
            continue #skip to next component
        fi
        IFS='+' read -r -a _COMPONENT_VERSION_ARG_ARR <<<"${!_COMPONENT_VERSION}"
        if [ -n "${_COMPONENT_VERSION_ARG_ARR[1]}" ]; then
            local _COMPONENT_VERSION_SUFFIX="+${_COMPONENT_VERSION_ARG_ARR[1]}"
        else
            local _COMPONENT_VERSION_SUFFIX=""
        fi
        IFS='.' read -r -a _COMPONENT_VERSION_ARR <<<"${!_COMPONENT_VERSION}"
        local _COMPONENT_VERSION_MAJOR=${_COMPONENT_VERSION_ARR[0]}
        local _COMPONENT_VERSION_MINOR=${_COMPONENT_VERSION_ARR[1]}
        local _COMPONENT_VERSION_PATCH=${_COMPONENT_VERSION_ARR[2]}
        local _COMPONENT_BRANCH="${_COMPONENT_VERSION_MAJOR}.${_COMPONENT_VERSION_MINOR}"
        local _COMPONENT_PATCH="${_COMPONENT_VERSION_MAJOR}.${_COMPONENT_VERSION_MINOR}.${_COMPONENT_VERSION_PATCH}"
        local _COMPONENT_ARCH="${ARCH_NAME}"
        local _COMPONENT_DISTRO="${DISTRO_NAME,,}${DISTRO_VERSION_MAJOR}${DISTRO_VERSION_MINOR}"
        if [[ "${_COMPONENT}" == "CUDNN" ]]; then
            if [[ "${!_COMPONENT_VERSION}" =~ "+cuda" ]]; then
                _CUDNN_CUDA_VERSION_MAJMIN="${!_COMPONENT_VERSION#*+cuda}"
            else
                _CUDNN_CUDA_VERSION_MAJMIN=${CUDA_VERSION_MAJMIN}
            fi
            _REPO_URL="https://urm.nvidia.com/artifactory/sw-gpu-cuda-installer-generic-local/packaging/v${_COMPONENT_BRANCH}_cuda_${_CUDNN_CUDA_VERSION_MAJMIN}/cudnn/linux-${ARCH_NAME_ALT}"
            _COMPONENT_VERSION_LATEST=$(curl -sL ${_REPO_URL} | sed -n 's/.*href=["'\'']\([^"'\'']*\).*/\1/p' | cut -f1 -d/ | sort -V | tail -n 1)
        fi
        if [[ "${_COMPONENT}" == "NCCL" ]]; then
            _NCCL_VALID_BRANCHES="master|stable|next"
            if [[ "${_COMPONENT_VERSION_PATCH}" =~ (${_NCCL_VALID_BRANCHES}) && -n ${_COMPONENT_VERSION_ARR[3]} ]]; then
                _NCCL_REPO_DIR="v${_COMPONENT_BRANCH}/NightlyBuilds/${_COMPONENT_VERSION_PATCH}"
                _REPO_URL="http://cuda-repo/release-candidates/Libraries/NCCL/${_NCCL_REPO_DIR}"
                _NCCL_NIGHTLY_LATEST=$(curl -sL ${_REPO_URL} | sed -n 's/.*href=["'\'']\([^"'\'']*\).*/\1/p' | cut -f1 -d/ | sort -V | tail -n 1 | cut -d"_" -f1)
                _COMPONENT_VERSION_LATEST=${_COMPONENT_VERSION_MAJOR}.${_COMPONENT_VERSION_MINOR}.${_COMPONENT_VERSION_PATCH}.${_NCCL_NIGHTLY_LATEST}
            else
                if [[ "${!_COMPONENT_VERSION}" =~ "+cuda" ]]; then
                    _NCCL_CUDA_VERSION_MAJMIN="${!_COMPONENT_VERSION#*+cuda}"
                else
                    _NCCL_CUDA_VERSION_MAJMIN=${CUDA_VERSION_MAJMIN}
                fi
                _DISTRO_STRING_UNDERSCORE="$(echo "$DISTRO_NAME" | awk '{print tolower($0)}')${DISTRO_VERSION_MAJOR}_${DISTRO_VERSION_MINOR}"
                _REPO_URL="https://urm.nvidia.com/artifactory/sw-gpu-cuda-installer-generic-local/packaging/nccl_stable_cuda${_NCCL_CUDA_VERSION_MAJMIN}/nccl/linux-${ARCH_NAME}/${_DISTRO_STRING_UNDERSCORE}"
                _COMPONENT_VERSION_LATEST=$(curl -sL ${_REPO_URL} | sed -n 's/.*href=["'\'']\([^"'\'']*\).*/\1/p' | cut -f1 -d/ | sort -V | tail -n 1)
            fi
        fi
        if [[ "${_COMPONENT}" == "TRT" ]]; then
            _REPO_URL="http://cuda-repo/release-candidates/Libraries/TensorRT/v${_COMPONENT_BRANCH}"
            _COMPONENT_VERSION_LATEST=$(curl -sL ${_REPO_URL} | sed 's/^.*>\([^<]*\)<.*$/\1/' | sed '/^$/d' | grep '^[0-9]' | sort -V | tail -1 | cut -d"-" -f1)
        fi
        if [[ "${_COMPONENT}" == "NSIGHT_SYSTEMS" ]]; then
            _REPO_URL="http://nsys/builds/Rel/CTK"
            _NSIGHT_SYSTEM_BUILD_FOLDER=$(curl -sL ${_REPO_URL} | sed -n 's/.*href=["'\'']\([^"'\'']*\).*/\1/p' | sed '/^$/d' | grep "^[0-9]" | sort -V | tail -n 1 | cut -f1 -d\/)
            _COMPONENT_VERSION_LATEST=$(curl -sL ${_REPO_URL}/${_NSIGHT_SYSTEM_BUILD_FOLDER} | sed -n 's/.*href=["'\'']\([^"'\'']*\).*/\1/p' | sed '/^$/d' | grep "^[0-9]" | sort -V | tail -n 1 | cut -f1 -d"-")
        fi
        echo "${_COMPONENT}_VERSION=${_COMPONENT_VERSION_LATEST}${_COMPONENT_VERSION_SUFFIX}"
    done
}

#####################################################
#### find the version of the component installed in the image
#####################################################
function get_component_version {
    # args
    local COMPONENT_NAME=$1
    declare -A COMPONENT_MAP=(["CUDA_DRIVER"]="cuda-compat"
        ["CUDA"]="cuda-cudart"
        ["CUBLAS"]="libcublas"
        ["CUFFT"]="libcufft"
        ["CURAND"]="libcurand"
        ["CUSPARSE"]="libcusparse"
        ["CUSOLVER"]="libcusolver"
        ["CUTENSOR"]="libcutensor"
        ["NPP"]="libnpp"
        ["NVJPEG"]="libnvjpeg"
        ["NCCL"]="libnccl"
        ["CUDNN"]="libcudnn"
        ["TRT"]="libnvinfer"
        ["NSIGHT_SYSTEMS"]="nsight-systems"
        ["NSIGHT_COMPUTE"]="nsight-compute"
    )
    local COMPONENT_VERSION=$(dpkg-query -W ${COMPONENT_MAP[${COMPONENT_NAME}]}* | cut -f 2 | cut -d"-" -f 1 | sort -n | tail -1)
    echo "${COMPONENT_VERSION}"
}

#####################################################
#### find component version in a kitpick version
#### takes KITPICK_VERSION and Package Name as inputs
#####################################################
function find_comp_ver_in_kitpick {
    # args
    local _KITPICK_VERSION=$1
    local _PACKAGE_NAME=$2
    IFS='.' read -r -a _KITPICK_VERSION <<<"${_KITPICK_VERSION}"
    if [ -z "${_KITPICK_VERSION[0]}" -o \
        -z "${_KITPICK_VERSION[1]}" -o \
        -z "${_KITPICK_VERSION[2]}" -o \
        -z "${_KITPICK_VERSION[3]}" ]; then
        echo "ERROR: KITPICK_VERSION should be of the form x.y.z.n"
        echo "Example: KITPICK_VERSION=11.0.3.007 where kitpick-label=11.0.3 & kitpick-candidate=007"
    fi
    local _KITPICK_MAJMIN=${_KITPICK_VERSION[0]}-${_KITPICK_VERSION[1]}
    if [[ "${DISTRO_NAME_ALT}" == "l4t" ]]; then _KITPICK_MAJMIN+="-tegra"; fi
    local _KITPICK_LABEL=${_KITPICK_VERSION[0]}.${_KITPICK_VERSION[1]}.${_KITPICK_VERSION[2]}
    local _KITPICK_CANDIDATE=${_KITPICK_VERSION[3]}
    local _KITPICK_SUMMARY_URL="http://cuda-repo/release-candidates/kitpicks/cuda-r${_KITPICK_MAJMIN}/${_KITPICK_LABEL}/${_KITPICK_CANDIDATE}/summary.txt"
    if ! url_exists "${_KITPICK_SUMMARY_URL}"; then
        echo "==========================" >&2
        echo "ERROR: unable to find the kitpick summary url for ${_KITPICK_SUMMARY_URL}" >&2
        echo "==========================" >&2
        return 1
    fi
    local _PACKAGE_VERSION=$(curl -sL ${_KITPICK_SUMMARY_URL} | grep ${_PACKAGE_NAME} | grep "[ ,]${ARCH_NAME}" | sed -e's/  */#/g' | cut -f3 -d"#")
    if [[ ! -n ${_PACKAGE_VERSION} ]]; then
        _PACKAGE_NAME=${_PACKAGE_NAME/driver-dev/cudart}
        _PACKAGE_NAME=${_PACKAGE_NAME/cccl/thrust}
        _PACKAGE_NAME=${_PACKAGE_NAME/nvvm/nvcc}
        _PACKAGE_NAME=${_PACKAGE_NAME/crt/nvcc}
        _PACKAGE_NAME=${_PACKAGE_NAME/toolkit-${_KITPICK_VERSION[0]}-config-common/cudart}
        _PACKAGE_NAME=${_PACKAGE_NAME/toolkit-${_KITPICK_VERSION[0]}-${_KITPICK_VERSION[1]}-config-common/cudart}
        _PACKAGE_NAME=${_PACKAGE_NAME/toolkit-config-common/cudart}
        _PACKAGE_NAME=${_PACKAGE_NAME/-dev/}
        _PACKAGE_NAME=${_PACKAGE_NAME//-/_}
        _PACKAGE_VERSION=$(curl -sL ${_KITPICK_SUMMARY_URL} | grep ${_PACKAGE_NAME} | grep "[ ,]${ARCH_NAME}" | sed -e's/  */#/g' | cut -f3 -d"#")
    fi
    echo "${_PACKAGE_VERSION}"

}

#####################################################
#### verify installed component version expected version
#### takes _COMP_VERSION and Package Name as inputs
#####################################################
function verify_installed_comp_ver {
    # args
    local _COMP_VERSION=$1
    local _PACKAGE_NAME=$2
    local _INSTALLED_VERSION=$(dpkg-query -W *${_PACKAGE_NAME}* | cut -f2 | sort -u | cut -f1 -d'-' | tail -1)
    if [[ "${_COMP_VERSION}" != "${_INSTALLED_VERSION}" ]]; then
        echo "==========================" >&2
        echo "ERROR: Installed version of ${_PACKAGE_NAME} is ${_INSTALLED_VERSION} while we are expecting ${_COMP_VERSION}" >&2
        echo "==========================" >&2
        echo 1
    else
        echo "==========================" >&2
        echo " Installed version of ${_PACKAGE_NAME} is ${_INSTALLED_VERSION} matches the expected version ${_COMP_VERSION}" >&2
        echo "==========================" >&2
        echo 0
    fi

}

#####################################################
#### VERSION NUMBER PARSING
#####################################################

### Parse distro version info

export DISTRO_NAME=$(cat /etc/os-release | awk -F= '/^NAME=/{print $2}' | tr -d '"' | cut -f1 -d' ')
_DISTRO_VERSION=$(grep ^VERSION_ID= /etc/os-release | cut -d \" -f 2)
IFS='.' read -r -a _DISTRO_VERSION <<<"${_DISTRO_VERSION}"
export DISTRO_VERSION_MAJOR=${_DISTRO_VERSION[0]}
export DISTRO_VERSION_MINOR=${_DISTRO_VERSION[1]}

if [ "${DISTRO_NAME}" != "Ubuntu" ] && [ "${DISTRO_NAME}" != "CentOS" ] && [ "${DISTRO_NAME}" != "AlmaLinux" ]; then
    echo "ERROR: Unsupported DISTRO_NAME $DISTRO_NAME"
fi

if [ -z "${DISTRO_VERSION_MINOR}" -a \
    "${DISTRO_NAME}" == "Ubuntu" ]; then
    echo "ERROR: Failed to parse ${DISTRO_NAME} field in /etc/os-release '${_DISTRO_VERSION}'"
    exit 1
fi

export DISTRO_VERSION_MAJMIN="${DISTRO_VERSION_MAJOR}.${DISTRO_VERSION_MINOR}"

if [ "${DISTRO_NAME}" == "CentOS" ] || [ "${DISTRO_NAME}" == "AlmaLinux" ]; then
    export DISTRO_NAME_ALT=rhel
fi

### Determine strings to use to represent this architecture

export ARCH_NAME=$(uname -m)

case "${ARCH_NAME}" in
x86_64)
    export ARCH_CUDA_REPO=x86_64
    export ARCH_NAME_ALT=x86_64
    export ARCH_NAME_ALT2=x86_64
    export ARCH_URL=x64
    export ARCH_URL_ALT=x64
    export ARCH_DEB=amd64
    ;;
aarch64)
    export ARCH_CUDA_REPO=arm64
    export ARCH_NAME_ALT=aarch64
    export ARCH_NAME_ALT2=sbsa
    export ARCH_URL=aarch64
    export ARCH_URL_ALT=aarch64sbsa
    export ARCH_DEB=arm64
    if [ -f /etc/ld.so.conf.d/nvidia-tegra.conf ]; then
        export DISTRO_NAME_ALT=l4t
        export DISTRO_OS_TYPE=mobile/target
        export ARCH_NAME_ALT2=arm64
    else
        export ARCH_NAME=sbsa
    fi
    ;;
*)
    echo "Unknown ARCH ${ARCH_NAME}"
    exit 1
    ;;
esac

### Parse CUDA version strings

# CUDA_VERSION should be like x.y.z
if [ -z "${CUDA_VERSION}" ]; then
    echo "ERROR: Missing CUDA_VERSION environment variable"
    exit 1
fi

IFS='.' read -r -a _CUDA_VERSION_ARR <<<"${CUDA_VERSION}"
export CUDA_VERSION_MAJOR=${_CUDA_VERSION_ARR[0]}
export CUDA_VERSION_MINOR=${_CUDA_VERSION_ARR[1]}
export CUDA_VERSION_BUILD=${_CUDA_VERSION_ARR[2]}

if [ -n "${_CUDA_VERSION_ARR[3]}" ]; then #if this field exists, requested CUDA_VERSION is a DVS build, .0 indicates automatics while [1-9] indicate virtual submissions
    export CUDA_VERSION_DVS_TYPE=${_CUDA_VERSION_ARR[3]}
fi
if [[ "${CUDA_VERSION}" == "gpgpu" ]]; then # CUDA_VERSION can be set to gpgpu to use builds from development branch "gpgpu"
    echo "Installing latest toolkit from CUDA Development Branch GPGPU"
    export CUDA_VERSION_MAJOR=0
    export CUDA_VERSION_MINOR=gpgpu
    export CUDA_VERSION_BUILD=0
elif
    [[ -z "${CUDA_VERSION_MAJOR}" ||
        -z "${CUDA_VERSION_MINOR}" ||
        -z "${CUDA_VERSION_BUILD}" ||
        -n "${_CUDA_VERSION_ARR[3]}" && "${CUDA_VERSION_BUILD}" -lt 10000000 && "${#_CUDA_VERSION_ARR[3]}" -ne 3 ||
        -n "${_CUDA_VERSION_ARR[3]}" && "${CUDA_VERSION_BUILD}" -gt 10000000 && "${#_CUDA_VERSION_ARR[3]}" -ne 1 ]] # kitpick_version should have last field of form 001..999
# dvs version should have CUDA_MAJOR.CUDA_MINOR.CHANGELIST.0..9 (0..9 is optional)
then
    echo "ERROR: CUDA_VERSION should be of the form x.y.z or x.y.<dvs-submission-id> or x.y.latest or x.y.z.<kitpick-candidate> or gpgpu"
    echo "Examples: CUDA_VERSION=11.0.221 for CUDA Toolkit build 11.0.221"
    echo "          CUDA_VERSION=11.0.28415548.0 or 11.0.28415548 for a DVS build at CL:28415548"
    echo "          CUDA_VERSION=11.0.28415548.1 for a DVS virtual build"
    echo "          CUDA_VERSION=11.0.3.007 for a kitpick label 11.0.3 candidate 007 http://cuda-repo/release-candidates/kitpicks/cuda-r11-0/11.0.3/007/"
    echo "          CUDA_VERSION=gpgpu for installing the latest toolkit build from CUDA Development Branch gpgpu"
    exit 1
fi

export CUDA_VERSION_MAJMIN=${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}
if [[ -n "${_CUDA_VERSION_ARR[3]}" && "${CUDA_VERSION_BUILD}" -lt 1000 && "${#_CUDA_VERSION_ARR[3]}" -ge 3 ]]; then
    export KITPICK_VERSION=${CUDA_VERSION}
    export CUDA_DRIVER_VERSION=$(find_comp_ver_in_kitpick "${KITPICK_VERSION}" nvidia_driver)
fi

if [[ "${DISTRO_NAME_ALT}" != "l4t" ]]; then

    # CUDA_DRIVER_VERSION should be like xxx.yy
    if [[ -z "${CUDA_DRIVER_VERSION}" ]]; then
        echo "ERROR: Missing CUDA_DRIVER_VERSION environment variable"
        exit 1
    fi

    IFS='.' read -r -a _CUDA_DRIVER_VERSION_ARR <<<"${CUDA_DRIVER_VERSION}"
    export CUDA_DRIVER_VERSION_MAJOR=${_CUDA_DRIVER_VERSION_ARR[0]}
    export CUDA_DRIVER_VERSION_MINOR=${_CUDA_DRIVER_VERSION_ARR[1]}
    export CUDA_DRIVER_VERSION_PATCH=${_CUDA_DRIVER_VERSION_ARR[2]} # may or may not have this one

    if [[ "${CUDA_DRIVER_VERSION}" == "bringup_h" || "${CUDA_DRIVER_VERSION}" == "cuda_a" ]]; then # CUDA_DRIVER_VERSION can be set to bringup_h or cuda_a to use builds from driver development branches
        echo "Installing latest driver from development branch ${CUDA_DRIVER_VERSION}"
    elif
        [ -z "${CUDA_DRIVER_VERSION_MAJOR}" -o \
            -z "${CUDA_DRIVER_VERSION_MINOR}" -o \
            -n "${_CUDA_DRIVER_VERSION_ARR[3]}" ]
    then
        echo "ERROR: CUDA_DRIVER_VERSION should be of the form xxx.yy for major release branches and xxx.yy.zz for side branches"
        exit 1
    fi

    case "${CUDA_DRIVER_VERSION_MAJOR}" in
    410) export CUDA_DRIVER_VERSION_BRANCH="400" ;;
    *) export CUDA_DRIVER_VERSION_BRANCH="${CUDA_DRIVER_VERSION_MAJOR}" ;;
    esac
fi

export ARTIFACTORY="https://cuda_ro:Nvidia3d!@urm.nvidia.com/artifactory/sw-gpu-cuda-installer-generic-local"

