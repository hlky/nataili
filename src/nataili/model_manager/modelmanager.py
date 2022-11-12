import hashlib
import json
import os
import shutil
import sys
import zipfile
from typing import Dict, List, Literal, Optional, Tuple, Union

import git
import requests
from tqdm import tqdm

from nataili.util import logger

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

pkg = importlib_resources.files("nataili")

models = json.load(open(pkg / "db.json"))
dependencies = json.load(open(pkg / "db_dep.json"))
remote_models = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db.json"
remote_dependencies = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db_dep.json"


class ModelManager:
    def __init__(self, model_base_path=None, hf_auth=None, download=True, disable_voodoo=True):
        if download:
            try:
                logger.init("Model Reference", status="Downloading")
                r = requests.get(remote_models)
                self.models = r.json()
                r = requests.get(remote_dependencies)
                self.dependencies = json.load(open(pkg / "db_dep.json"))
                logger.init_ok("Model Reference", status="OK")
            except Exception:
                logger.init_err("Model Reference", status="Download Error")
                self.models = json.load(open(pkg / "db.json"))
                self.dependencies = json.load(open(pkg / "db_dep.json"))
                logger.init_warn("Model Reference", status="Local")
        else:
            self.models = json.load(open(pkg / "db.json"))
            self.dependencies = json.load(open(pkg / "db_dep.json"))
        self.available_models = []
        self.tainted_models = []
        self.available_dependencies = []
        self.loaded_models = {}
        self.hf_auth = None
        self.set_authentication(hf_auth)
        self.disable_voodoo = disable_voodoo
        self.model_base_path = model_base_path if model_base_path else os.path.join(os.path.expanduser('~'), '.nataili')
        if not os.path.exists(self.model_base_path):
            os.makedirs(self.model_base_path)

    def init(self):
        dependencies_available = []
        for dependency in self.dependencies:
            if self.check_available(self.get_dependency_files(dependency)):
                dependencies_available.append(dependency)
        self.available_dependencies = dependencies_available

        models_available = []
        for model in self.models:
            if self.check_available(self.get_model_files(model)):
                models_available.append(model)
        self.available_models = models_available

        if self.hf_auth is not None:
            if "username" not in self.hf_auth and "password" not in self.hf_auth:
                raise ValueError("hf_auth must contain username and password")
            else:
                if self.hf_auth["username"] == "" or self.hf_auth["password"] == "":
                    raise ValueError("hf_auth must contain username and password")
        return True

    def set_authentication(self, hf_auth=None):
        # We do not let No authentication override previously set auth
        if not hf_auth and self.hf_auth:
            return
        self.hf_auth = hf_auth
        if hf_auth:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_auth.get("password")

    def has_authentication(self):
        if self.hf_auth:
            return True
        return False

    def get_model(self, model_name):
        return self.models.get(model_name)

    def get_filtered_models(self, **kwargs):
        """Get all model names.
        Can filter based on metadata of the model reference db
        """
        filtered_models = self.models
        for keyword in kwargs:
            iterating_models = filtered_models.copy()
            filtered_models = {}
            for model in iterating_models:
                # logger.debug([keyword,iterating_models[model].get(keyword),kwargs[keyword]])
                if iterating_models[model].get(keyword) == kwargs[keyword]:
                    filtered_models[model] = iterating_models[model]
        return filtered_models

    def get_filtered_model_names(self, **kwargs):
        filtered_models = self.get_filtered_models(**kwargs)
        return list(filtered_models.keys())

    def get_dependency(self, dependency_name):
        return self.dependencies[dependency_name]

    def get_model_files(self, model_name):
        if self.models[model_name]["type"] == "diffusers":
            return []
        return self.models[model_name]["config"]["files"]

    def get_dependency_files(self, dependency_name):
        return self.dependencies[dependency_name]["config"]["files"]

    def get_model_download(self, model_name):
        return self.models[model_name]["config"]["download"]

    def get_dependency_download(self, dependency_name):
        return self.dependencies[dependency_name]["config"]["download"]

    def get_available_models(self):
        return self.available_models

    def get_available_dependencies(self):
        return self.available_dependencies

    def get_loaded_models(self):
        return self.loaded_models

    def get_loaded_models_names(self):
        return list(self.loaded_models.keys())

    def get_loaded_model(self, model_name):
        return self.loaded_models[model_name]

    def unload_model(self, model_name):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            return True
        return False

    def unload_all_models(self):
        for model in self.loaded_models:
            del self.loaded_models[model]
        return True

    def taint_model(self, model_name):
        """Marks a model as not valid by remiving it from available_models"""
        if model_name in self.available_models:
            self.available_models.remove(model_name)
            self.tainted_models.append(model_name)

    def taint_models(self, models):
        for model in models:
            self.taint_model(model)

    def validate_model(self, model_name):
        files = self.get_model_files(model_name)
        for file_details in files:
            if not self.check_file_available(file_details["path"]):
                return False
            if not self.validate_file(file_details):
                return False
        return True

    def validate_file(self, file_details):
        if "md5sum" in file_details:
            file_name = file_details["path"]
            logger.debug(f"Getting md5sum of {file_name}")
            with open(file_name, "rb") as file_to_check:
                file_hash = hashlib.md5()
                while chunk := file_to_check.read(8192):
                    file_hash.update(chunk)
            if file_details["md5sum"] != file_hash.hexdigest():
                return False
        return True

    def check_file_available(self, file_path):
        return os.path.exists(os.path.join(self.model_base_path, file_path))

    def check_available(self, files):
        available = True
        for file in files:
            if not self.check_file_available(file["path"]):
                available = False
        return available

    def download_file(self, url, file_path):
        # make directory
        file_path = os.path.join(self.model_base_path, file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pbar_desc = file_path.split("/")[-1]
        r = requests.get(url, stream=True, allow_redirects=True)
        with open(file_path, "wb") as f:
            with tqdm(
                # all optional kwargs
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=pbar_desc,
                total=int(r.headers.get("content-length", 0)),
            ) as pbar:
                for chunk in r.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def download_model(self, model_name):
        if model_name in self.available_models:
            logger.info(f"{model_name} is already available.")
            return True
        download = self.get_model_download(model_name)
        files = self.get_model_files(model_name)
        for i in range(len(download)):
            file_path = (
                f"{download[i]['file_path']}/{download[i]['file_name']}"
                if "file_path" in download[i]
                else files[i]["path"]
            )
            file_path = os.path.join(self.model_base_path, file_path)

            if "file_url" in download[i]:
                download_url = download[i]["file_url"]
                if "hf_auth" in download[i]:
                    username = self.hf_auth["username"]
                    password = self.hf_auth["password"]
                    download_url = download_url.format(username=username, password=password)
            if "file_name" in download[i]:
                download_name = download[i]["file_name"]
            if "file_path" in download[i]:
                download_path = download[i]["file_path"]
                download_path = os.path.join(self.model_base_path, download_path)

            if "manual" in download[i]:
                logger.warning(
                    f"The model {model_name} requires manual download from {download_url}. "
                    f"Please place it in {download_path}/{download_name} then press ENTER to continue..."
                )
                input("")
                continue
            # TODO: simplify
            if "file_content" in download[i]:
                file_content = download[i]["file_content"]
                logger.info(f"writing {file_content} to {file_path}")
                # make directory download_path
                os.makedirs(download_path, exist_ok=True)
                # write file_content to download_path/download_name
                with open(os.path.join(download_path, download_name), "w") as f:
                    f.write(file_content)
            elif "symlink" in download[i]:
                logger.info(f"symlink {file_path} to {download[i]['symlink']}")
                symlink = download[i]["symlink"]
                symlink = os.path.join(self.model_base_path, symlink)
                # make directory symlink
                os.makedirs(download_path, exist_ok=True)
                # make symlink from download_path/download_name to symlink
                os.symlink(symlink, os.path.join(download_path, download_name))
            elif "git" in download[i]:
                logger.info(f"git clone {download_url} to {file_path}")
                # make directory download_path
                os.makedirs(file_path, exist_ok=True)
                git.Git(file_path).clone(download_url)
                if "post_process" in download[i]:
                    for post_process in download[i]["post_process"]:
                        if "delete" in post_process:
                            post_process['delete'] = os.path.join(self.model_base_path, post_process['delete'])
                            # delete folder post_process['delete']
                            logger.info(f"delete {post_process['delete']}")
                            try:
                                shutil.rmtree(post_process["delete"])
                            except PermissionError as e:
                                logger.error(
                                    f"[!] Something went wrong while deleting the `{post_process['delete']}`. "
                                    "Please delete it manually."
                                )
                                logger.error("PermissionError: ", e)
            else:
                if not self.check_file_available(file_path) or model_name in self.tainted_models:
                    logger.debug(f"Downloading {download_url} to {file_path}")
                    self.download_file(download_url, file_path)
        if not self.validate_model(model_name):
            return False
        if model_name in self.tainted_models:
            self.tainted_models.remove(model_name)
        self.init()
        return True

    def download_dependency(self, dependency_name):
        if dependency_name in self.available_dependencies:
            logger.info(f"{dependency_name} is already installed.")
            return True
        download = self.get_dependency_download(dependency_name)
        files = self.get_dependency_files(dependency_name)
        for i in range(len(download)):
            if "git" in download[i]:
                logger.warning("git download not implemented yet")
                break

            file_path = files[i]["path"]
            file_path = os.path.join(self.model_base_path, file_path)
            if "file_url" in download[i]:
                download_url = download[i]["file_url"]
            if "file_name" in download[i]:
                download_name = download[i]["file_name"]
            if "file_path" in download[i]:
                download_path = download[i]["file_path"]
                download_path = os.path.join(self.model_base_path, download_path)
            logger.debug(download_name)
            if "unzip" in download[i]:
                zip_path = f"temp/{download_name}.zip"
                zip_path = os.path.join(self.model_base_path, zip_path)
                # os dirname zip_path
                # mkdir temp
                os.makedirs(os.path.join(self.model_base_path, "temp"), exist_ok=True)

                self.download_file(download_url, zip_path)
                logger.info(f"unzip {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall("temp/")
                # move temp/sd-concepts-library-main/sd-concepts-library to download_path
                logger.info(f"move temp/{download_name}-main/{download_name} to {download_path}")
                shutil.move(f"{self.model_base_path}/temp/{download_name}-main/{download_name}", download_path)
                logger.info(f"delete {zip_path}")
                os.remove(zip_path)
                logger.info(f"delete temp/{download_name}-main/")
                shutil.rmtree(f"{self.model_base_path}/temp/{download_name}-main")
            else:
                if not self.check_file_available(file_path):
                    logger.init(f"{file_path}", status="Downloading")
                    self.download_file(download_url, file_path)
        self.init()
        return True

    def download_all_models(self):
        for model in self.get_filtered_model_names(download_all=True):
            if not self.check_model_available(model):
                logger.init(f"{model}", status="Downloading")
                self.download_model(model)
            else:
                logger.info(f"{model} is already downloaded.")
        return True

    def download_all_dependencies(self):
        for dependency in self.dependencies:
            if not self.check_dependency_available(dependency):
                logger.init(f"{dependency}", status="Downloading")
                self.download_dependency(dependency)
            else:
                logger.info(f"{dependency} is already installed.")
        return True

    def download_all(self):
        self.download_all_dependencies()
        self.download_all_models()
        return True

    def check_model_available(self, model_name):
        if model_name not in self.models:
            return False
        return self.check_available(self.get_model_files(model_name))

    def check_dependency_available(self, dependency_name):
        if dependency_name not in self.dependencies:
            return False
        return self.check_available(self.get_dependency_files(dependency_name))

    def check_all_available(self):
        for model in self.models:
            if not self.check_model_available(model):
                return False
        for dependency in self.dependencies:
            if not self.check_dependency_available(dependency):
                return False
        return True
