#!/usr/bin/env python3
"""
AWS SQS를 통한 SD1.5 LoRA 학습 핸들러
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Dict, Any
import traceback
import time
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

# .env 파일 로딩
load_dotenv()

SQS_URL_LORA_TRAINING = os.getenv('SQS_URL_LORA_TRAINING')
FIREBASE_SERVICE_ACCOUNT_KEY = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
FIREBASE_DATABASE_URL = os.getenv('FIREBASE_DATABASE_URL')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoRATrainingHandler:
    def __init__(self, queue_url: str, region_name: str = 'ap-northeast-2', identifier: str = 'shs', 
                 tmp_dir: str = './work_dir', sqs_max_messages: int = 10, 
                 sqs_visibility_timeout: int = 3, sqs_wait_time: int = 20,
                 s3_bucket_name: str = None, s3_region: str = None, virtual_env_bin_path: str = None):
        """
        SQS LoRA 학습 핸들러 초기화
        
        Args:
            queue_url: SQS 큐 URL
            region_name: AWS 리전
            identifier: 식별자
            tmp_dir: 작업 디렉토리 (기본값: './work_dir')
            sqs_max_messages: SQS 최대 메시지 수 (기본값: 10)
            sqs_visibility_timeout: SQS 가시성 타임아웃 (기본값: 3)
            sqs_wait_time: SQS 대기 시간 (기본값: 20)
            s3_bucket_name: S3 버킷 이름
            s3_region: S3 리전 (기본값: SQS와 동일한 리전)
            virtual_env_bin_path: conda 환경명 (기본값: 현재 환경)
        """
        self.queue_url = queue_url
        self.region_name = region_name
        self.sqs = None
        self.s3 = None
        self.running_processes = set()
        self.identifier = identifier
        self.tmp_dir = tmp_dir
        self.sqs_max_messages = sqs_max_messages
        self.sqs_visibility_timeout = sqs_visibility_timeout
        self.sqs_wait_time = sqs_wait_time
        self.s3_bucket_name = s3_bucket_name
        self.s3_region = s3_region or region_name
        self.virtual_env_bin_path = virtual_env_bin_path
        self.firebase_initialized = False
        
    def _get_aws_credentials(self) -> tuple:
        """
        .env 파일에서 AWS 자격 증명 읽기
        
        Returns:
            (aws_access_key_id, aws_secret_access_key) 튜플
        """
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not aws_access_key_id or not aws_secret_access_key:
            logger.error("AWS 자격 증명이 .env 파일에 설정되지 않았습니다. AWS_ACCESS_KEY_ID와 AWS_SECRET_ACCESS_KEY를 확인하세요.")
            sys.exit(1)
        
        return aws_access_key_id, aws_secret_access_key
        
    def _init_sqs_client(self):
        """SQS 클라이언트 초기화"""
        try:
            aws_access_key_id, aws_secret_access_key = self._get_aws_credentials()
            
            self.sqs = boto3.client(
                'sqs',
                region_name=self.region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            logger.info(f"SQS 클라이언트 초기화 완료 - 리전: {self.region_name}")
        except NoCredentialsError:
            logger.error("AWS 자격 증명을 찾을 수 없습니다. .env 파일의 AWS 설정을 확인하세요.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"SQS 클라이언트 초기화 실패: {e}")
            sys.exit(1)
    
    def _init_s3_client(self):
        """S3 클라이언트 초기화"""
        try:
            aws_access_key_id, aws_secret_access_key = self._get_aws_credentials()
            
            self.s3 = boto3.client(
                's3',
                region_name=self.s3_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            logger.info(f"S3 클라이언트 초기화 완료 - 리전: {self.s3_region}")
        except NoCredentialsError:
            logger.error("AWS 자격 증명을 찾을 수 없습니다. .env 파일의 AWS 설정을 확인하세요.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"S3 클라이언트 초기화 실패: {e}")
            sys.exit(1)
    
    def _init_firebase_client(self):
        """Firebase 클라이언트 초기화"""
        try:
            if self.firebase_initialized:
                return
            
            if not FIREBASE_SERVICE_ACCOUNT_KEY:
                logger.error("FIREBASE_SERVICE_ACCOUNT_KEY 환경변수가 설정되지 않았습니다.")
                sys.exit(1)
                
            if not FIREBASE_DATABASE_URL:
                logger.error("FIREBASE_DATABASE_URL 환경변수가 설정되지 않았습니다.")
                sys.exit(1)
            
            # Firebase Admin SDK 초기화
            if not firebase_admin._apps:
                # 서비스 계정 키 파일 경로 또는 딕셔너리 확인
                if os.path.isfile(FIREBASE_SERVICE_ACCOUNT_KEY):
                    # 파일 경로인 경우
                    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY)
                else:
                    # JSON 문자열인 경우
                    service_account_info = json.loads(FIREBASE_SERVICE_ACCOUNT_KEY)
                    cred = credentials.Certificate(service_account_info)
                
                firebase_admin.initialize_app(cred, {
                    'databaseURL': FIREBASE_DATABASE_URL
                })
                logger.info("Firebase Admin SDK 초기화 완료")
            
            self.firebase_initialized = True
            
        except json.JSONDecodeError:
            logger.error("FIREBASE_SERVICE_ACCOUNT_KEY JSON 형식이 올바르지 않습니다.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Firebase 클라이언트 초기화 실패: {e}")
            sys.exit(1)
    
    def _update_training_status(self, request_id: str, status: str, error_msg: str = None, **kwargs):
        """
        Firebase Realtime Database에 학습 상태 업데이트
        
        Args:
            request_id: 요청 ID
            status: 상태 (REQUESTED, TRAINING, SUCCESS, FAILED)
            error_msg: 에러 메시지 (실패 시)
            **kwargs: 추가 필드
        """
        try:
            if not self.firebase_initialized:
                self._init_firebase_client()
            
            # UTC 타임스탬프 생성
            timestamp = int(time.time())
            
            # 상태 데이터 구성
            status_data = {
                'status': status,
                'timestamp': timestamp,
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            # 에러 메시지가 있으면 추가
            if error_msg:
                status_data['error_msg'] = error_msg
            
            # 추가 필드들 병합
            status_data.update(kwargs)
            
            # Firebase에 상태 업데이트
            ref = db.reference(f'train_status/{request_id}')
            ref.set(status_data)
            
            logger.info(f"학습 상태 업데이트 완료 - request_id: {request_id}, status: {status}")
            
        except Exception as e:
            logger.error(f"학습 상태 업데이트 실패: {e}")
            # Firebase 상태 업데이트 실패해도 학습은 계속 진행
    
    def _get_training_status(self, request_id: str) -> Dict[str, Any]:
        """
        Firebase Realtime Database에서 학습 상태 조회
        
        Args:
            request_id: 요청 ID
            
        Returns:
            상태 데이터 딕셔너리
        """
        try:
            if not self.firebase_initialized:
                self._init_firebase_client()
            
            ref = db.reference(f'train_status/{request_id}')
            status_data = ref.get()
            
            return status_data or {}
            
        except Exception as e:
            logger.error(f"학습 상태 조회 실패: {e}")
            return {}
    
    def _create_directory_structure(self, request_id: str) -> Dict[str, str]:
        """
        학습용 디렉토리 구조 생성
        
        Args:
            request_id: 요청 ID
            
        Returns:
            생성된 디렉토리 경로들 딕셔너리
        """
        base_dir = os.path.join(self.tmp_dir, request_id)
        img_dir = os.path.join(base_dir, 'img', f'1_{self.identifier} woman')
        log_dir = os.path.join(base_dir, 'log')
        model_dir = os.path.join(base_dir, 'model')
        
        # 디렉토리 생성
        directories = [img_dir, log_dir, model_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"디렉토리 생성: {directory}")
        
        return {
            'base_dir': base_dir,
            'img_dir': img_dir,
            'log_dir': log_dir,
            'model_dir': model_dir
        }
    
    def _download_s3_images(self, s3_folder_path: str, local_img_dir: str) -> list:
        """
        S3에서 이미지 파일들을 다운로드
        
        Args:
            s3_folder_path: S3 폴더 경로 (루트 경로부터, 예: folder/subfolder/path)
            local_img_dir: 로컬 이미지 디렉토리 경로
            
        Returns:
            다운로드된 이미지 파일 목록
        """
        if not self.s3:
            self._init_s3_client()
        
        if not self.s3_bucket_name:
            raise ValueError("S3 버킷 이름이 설정되지 않았습니다.")
        
        # S3 경로에서 앞/뒤 슬래시 제거
        folder_path = s3_folder_path.strip('/')
        
        logger.info(f"S3에서 이미지 다운로드 시작: s3://{self.s3_bucket_name}/{folder_path} -> {local_img_dir}")
        
        downloaded_files = []
        
        try:
            # S3 객체 목록 가져오기
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket_name,
                Prefix=folder_path
            )
            
            if 'Contents' not in response:
                logger.warning(f"S3 경로에 파일이 없습니다: s3://{self.s3_bucket_name}/{folder_path}")
                return downloaded_files
            
            # 이미지 파일만 필터링
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
            
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                
                # 파일명이 없는 경우 (디렉토리인 경우) 건너뛰기
                if not filename:
                    continue
                
                # 파일 확장자 확인
                _, ext = os.path.splitext(filename.lower())
                if ext not in image_extensions:
                    continue
                
                # 로컬 파일 경로
                local_file_path = os.path.join(local_img_dir, filename)
                
                # 파일 다운로드
                try:
                    self.s3.download_file(self.s3_bucket_name, key, local_file_path)
                    downloaded_files.append(filename)
                    logger.info(f"파일 다운로드 완료: {filename}")
                except Exception as e:
                    logger.error(f"파일 다운로드 실패 {filename}: {e}")
            
            logger.info(f"총 {len(downloaded_files)}개 이미지 파일 다운로드 완료")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"S3 이미지 다운로드 중 오류: {e}")
            raise
    
    def _create_caption_files(self, img_dir: str, image_files: list, identifier: str, hair_length: str):
        """
        이미지 파일과 동일한 이름의 캡션 txt 파일들을 생성
        
        Args:
            img_dir: 이미지 디렉토리 경로
            image_files: 이미지 파일 목록
            identifier: 식별자
            hair_length: 머리 길이
        """
        logger.info(f"캡션 파일 생성 시작: {len(image_files)}개 파일")

        if "hair" not in hair_length:
            hair_length = f"{hair_length} hair"
        
        caption_content = f"{identifier}, {hair_length}"
        
        for image_file in image_files:
            # 이미지 파일명에서 확장자 제거
            name_without_ext = os.path.splitext(image_file)[0]
            
            # txt 파일 경로
            txt_file_path = os.path.join(img_dir, f"{name_without_ext}.txt")
            
            try:
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    f.write(caption_content)
                logger.info(f"캡션 파일 생성 완료: {name_without_ext}.txt")
            except Exception as e:
                logger.error(f"캡션 파일 생성 실패 {name_without_ext}.txt: {e}")
        
        logger.info("모든 캡션 파일 생성 완료")
    
    def _build_training_command(self, message_data: Dict[str, Any]) -> list:
        """
        학습 명령어 구성 및 데이터 준비
        
        Args:
            message_data: SQS 메시지 데이터
            
        Returns:
            학습 명령어 리스트
        """
        # 새로운 S3 메시지 구조의 필수 파라미터
        required_params = [
            'request_id',
            's3_folder_path',
            'style_name',
            'style_type',
            'epoch',
            'network_alpha',
            'network_dim',
            'hair_length'
        ]
        
        # 필수 파라미터 검증
        for param in required_params:
            if param not in message_data:
                raise ValueError(f"필수 파라미터 누락: {param}")
        
        request_id = message_data['request_id']
        s3_folder_path = message_data['s3_folder_path']
        hair_length = message_data['hair_length']
        
        # 1. 디렉토리 구조 생성
        dirs = self._create_directory_structure(request_id)
        
        # 2. S3에서 이미지 다운로드
        image_files = self._download_s3_images(s3_folder_path, dirs['img_dir'])
        
        if not image_files:
            raise ValueError(f"다운로드된 이미지가 없습니다: {s3_folder_path}")
        
        # 3. 캡션 파일 생성
        self._create_caption_files(dirs['img_dir'], image_files, self.identifier, hair_length)
        
        # 4. 기본 명령어 구성
        if self.virtual_env_bin_path:
            # conda 환경이 지정된 경우
            if os.name == 'nt':  # Windows
                cmd = [
                    'cmd', '/c',
                    f'{self.virtual_env_bin_path}/accelerate launch --num_processes=1 --num_cpu_threads_per_process=2 ./train_network.py'
                ]
            else:  # Linux/Mac
                # cmd = [
                #     'bash', '-c',
                #     f'{self.virtual_env_bin_path}/accelerate launch --num_processes=1 --num_cpu_threads_per_process=2 ./train_network.py'
                # ]
                cmd = [
                f'{self.virtual_env_bin_path}/accelerate', 'launch',
                '--num_processes=1',
                '--num_cpu_threads_per_process=2',
                './train_network.py'
            ]
        else:
            # 현재 환경 사용
            cmd = [
                'accelerate', 'launch',
                '--num_processes=1',
                '--num_cpu_threads_per_process=2',
                './train_network.py'
            ]

        if "hair" not in hair_length:
            hair_length = f"{hair_length} hair"
        
        # 5. 기본 파라미터 설정
        default_params = {
            'enable_bucket': True,
            'min_bucket_reso': 512,
            'max_bucket_reso': 640,
            'resolution': '512,640',
            'save_model_as': 'safetensors',
            'network_module': 'networks.lora',
            'text_encoder_lr': 5e-05,
            'unet_lr': 0.0001,
            'lr_scheduler_num_cycles': 1,
            'no_half_vae': True,
            'learning_rate': 0.0001,
            'lr_scheduler': 'cosine',
            'lr_warmup_steps': 46,
            'train_batch_size': 1,
            'save_every_n_epochs': 50,
            'mixed_precision': 'fp16',
            'save_precision': 'fp16',
            'cache_latents': True,
            'optimizer_type': 'AdamW8bit',
            'max_data_loader_n_workers': 0,
            'bucket_reso_steps': 64,
            # 'xformers': True,
            'bucket_no_upscale': True,
            'noise_offset': 0.0,
            'sample_every_n_epochs': 50,
            'sample_prompts': f'{self.identifier}, {hair_length}',
            'sample_sampler': 'k_dpm_2',
            # 고정 파라미터
            'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
            'train_data_dir': os.path.dirname(dirs['img_dir']),
            'output_dir': dirs['model_dir'],
            'output_name': message_data['style_name'],
            'logging_dir': dirs['log_dir']
        }
        
        # 6. 메시지 데이터에서 오버라이드할 파라미터 설정
        override_params = {
            'max_train_epochs': int(message_data['epoch']),
            'network_alpha': int(message_data['network_alpha']),
            'network_dim': int(message_data['network_dim'])
        }
        
        # 파라미터 병합
        params = {**default_params, **override_params}
        
        # 7. 명령어에 파라미터 추가
        if self.virtual_env_bin_path and os.name == 'nt':
            # Windows에서 conda 환경 사용 시 - 명령어 문자열에 파라미터 추가
            cmd_parts = []
            for key, value in params.items():
                if key in ['enable_bucket', 'no_half_vae', 'cache_latents', 'xformers', 'bucket_no_upscale']:
                    if value:
                        cmd_parts.append(f'--{key}')
                else:
                    # 경로를 정규화하여 슬래시 사용
                    if 'dir' in key.lower() and isinstance(value, str):
                        value = value.replace('\\', '/')
                    cmd_parts.extend([f'--{key}', f'{str(value)}'])
            
            # 기존 명령어에 파라미터 추가
            cmd[2] = cmd[2] + ' ' + ' '.join(cmd_parts)
        else:
            # 일반적인 경우
            for key, value in params.items():
                if key in ['enable_bucket', 'no_half_vae', 'cache_latents', 'xformers', 'bucket_no_upscale']:
                    if value:
                        cmd.append(f'--{key}')
                else:
                    # 경로를 정규화하여 슬래시 사용
                    if 'dir' in key.lower() and isinstance(value, str):
                        value = value.replace('\\', '/')
                    cmd.extend([f'--{key}', f'{str(value)}'])
        
        logger.info(f"학습 데이터 준비 완료 - 이미지: {len(image_files)}개, 출력: {dirs['model_dir']}")
        return cmd, dirs
    
    async def _run_training_async(self, cmd: list, output_name: str, message_data: Dict[str, Any], dirs: Dict[str, str]):
        """
        비동기 학습 실행
        
        Args:
            cmd: 실행할 명령어 리스트
            output_name: 출력 모델 이름
            message_data: 원본 메시지 데이터 (후작업용)
            dirs: 디렉토리 정보
        """
        process = None
        try:
            logger.info(f"학습 시작: {output_name}")
            logger.info(f"실행 명령어: {' '.join(cmd)}")
            
            # 학습 시작 시 TRAINING 상태로 업데이트
            request_id = message_data.get('request_id')
            if request_id:
                self._update_training_status(
                    request_id, 
                    'TRAINING', 
                    style_name=message_data.get('style_name', 'Unknown'),
                    training_started_at=datetime.utcnow().isoformat() + 'Z'
                )
            
            # 로그 파일 경로 설정 - base_dir/train.log
            log_file = os.path.join(dirs['base_dir'], 'train.log')
            
            # 비동기 프로세스 실행
            with open(log_file, 'w', encoding='utf-8') as log:
                # Windows에서 UTF-8 인코딩 환경변수 설정
                env = os.environ.copy()
                # GPU 0번 사용 설정
                env['CUDA_VISIBLE_DEVICES'] = '0'
                if os.name == 'nt':
                    env['PYTHONIOENCODING'] = 'utf-8'
                    env['CHCP'] = '65001'
                
                if self.virtual_env_bin_path and os.name == 'nt':
                    # Windows에서 conda 환경 사용 시 - UTF-8 인코딩으로 실행
                    # CMD를 UTF-8 모드로 시작하고 conda 명령어 실행
                    utf8_cmd = f'chcp 65001 >nul && {cmd[2]}'
                    process = await asyncio.create_subprocess_shell(
                        utf8_cmd,
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=os.getcwd(),
                        env=env
                    )
                elif os.name == 'nt':
                    # Windows에서 일반적인 경우도 UTF-8 인코딩 적용
                    # accelerate 명령어를 shell로 실행하되 UTF-8 설정 추가
                    cmd_str = ' '.join(cmd)
                    utf8_cmd = f'chcp 65001 >nul && {cmd_str}'
                    process = await asyncio.create_subprocess_shell(
                        utf8_cmd,
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=os.getcwd(),
                        env=env
                    )
                else:
                    # Linux/Mac의 경우
                    logger.info(f"cmd: {cmd}")
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=os.getcwd(),
                        env=env
                    )
                logger.info(f"process: {process}")

                self.running_processes.add(process)
                logger.info(f"학습 프로세스 시작됨 - PID: {process.pid}, 로그: {log_file}")
                
                # 프로세스 완료 대기
                await process.wait()
                logger.info(f"process.returncode: {process.returncode}")
            
            # 로그 파일에서 실제 에러 확인
            if process.returncode == 0:
                logger.info(f"학습 완료: {output_name}")
                
                # 학습 성공 후작업 실행
                await self._post_training_success(output_name, message_data, log_file)
            else:
                logger.error(f"학습 실패: {output_name}, 종료 코드: {process.returncode}")
                
                # 로그 파일에서 에러 정보 읽기
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                        logger.error(f"학습 에러 로그:\n{log_content[-2000:]}")  # 마지막 2000자만 로그
                except Exception as e:
                    logger.error(f"로그 파일 읽기 실패: {e}")
                
                # 학습 실패 후작업 실행
                await self._post_training_failure(output_name, message_data, log_file, process.returncode)
                
        except Exception as e:
            logger.error(f"학습 실행 중 오류 발생: {e}")
            
            # 예외 발생 후작업 실행
            await self._post_training_error(output_name, message_data, str(e))
        finally:
            if process and process in self.running_processes:
                self.running_processes.remove(process)
    
    async def _post_training_success(self, output_name: str, message_data: Dict[str, Any], log_file: str):
        """
        학습 성공 후작업
        
        Args:
            output_name: 출력 모델 이름
            message_data: 원본 메시지 데이터
            log_file: 학습 로그 파일 경로
        """
        try:
            logger.info(f"학습 성공 후작업 시작: {output_name}")
            
            # SUCCESS 상태로 업데이트
            request_id = message_data.get('request_id')
            if request_id:
                # 로그에서 최종 손실값 추출
                final_loss = await self._extract_final_loss_from_log(log_file)
                
                self._update_training_status(
                    request_id, 
                    'SUCCESS', 
                    style_name=message_data.get('style_name', 'Unknown'),
                    training_completed_at=datetime.utcnow().isoformat() + 'Z',
                    final_loss=final_loss
                )
            
            # 성공 콜백 실행
            callback_url = message_data.get('callback_url')
            if callback_url:
                await self._send_completion_callback(callback_url, {
                    'status': 'success',
                    'output_name': output_name,
                    'log_file': log_file
                })
            
            logger.info(f"학습 성공 후작업 완료: {output_name}")
            
        except Exception as e:
            logger.error(f"후작업 실행 중 오류: {e}")
    
    async def _post_training_failure(self, output_name: str, message_data: Dict[str, Any], 
                                   log_file: str, return_code: int):
        """
        학습 실패 후작업
        
        Args:
            output_name: 출력 모델 이름
            message_data: 원본 메시지 데이터
            log_file: 학습 로그 파일 경로
            return_code: 프로세스 종료 코드
        """
        try:
            logger.info(f"학습 실패 후작업 시작: {output_name}")
            
            # 1. 에러 로그에서 실패 원인 추출
            error_info = await self._extract_error_from_log(log_file)
            
            # 2. FAILED 상태로 업데이트
            request_id = message_data.get('request_id')
            if request_id:
                self._update_training_status(
                    request_id, 
                    'FAILED', 
                    error_msg=error_info,
                    style_name=message_data.get('style_name', 'Unknown'),
                    training_failed_at=datetime.utcnow().isoformat() + 'Z',
                    return_code=return_code
                )
            
            # 3. 실패 콜백 실행
            callback_url = message_data.get('callback_url')
            if callback_url:
                await self._send_completion_callback(callback_url, {
                    'status': 'failed',
                    'output_name': output_name,
                    'return_code': return_code,
                    'error_info': error_info,
                    'log_file': log_file
                })
            
            logger.info(f"학습 실패 후작업 완료: {output_name}")
            
        except Exception as e:
            logger.error(f"실패 후작업 실행 중 오류: {e}")
    
    async def _post_training_error(self, output_name: str, message_data: Dict[str, Any], error_msg: str):
        """
        학습 예외 발생 후작업
        
        Args:
            output_name: 출력 모델 이름
            message_data: 원본 메시지 데이터
            error_msg: 에러 메시지
        """
        try:
            logger.info(f"학습 예외 후작업 시작: {output_name}")
            
            # FAILED 상태로 업데이트
            request_id = message_data.get('request_id')
            if request_id:
                self._update_training_status(
                    request_id, 
                    'FAILED', 
                    error_msg=error_msg,
                    style_name=message_data.get('style_name', 'Unknown'),
                    training_failed_at=datetime.utcnow().isoformat() + 'Z'
                )
            
            # 예외 콜백 실행
            callback_url = message_data.get('callback_url')
            if callback_url:
                await self._send_completion_callback(callback_url, {
                    'status': 'error',
                    'output_name': output_name,
                    'error_message': error_msg
                })
            
            logger.info(f"학습 예외 후작업 완료: {output_name}")
            
        except Exception as e:
            logger.error(f"예외 후작업 실행 중 오류: {e}")
    
    async def _extract_final_loss_from_log(self, log_file: str) -> str:
        """로그 파일에서 최종 손실값 추출"""
        try:
            if not os.path.exists(log_file):
                return None
                
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 마지막 몇 줄에서 loss 정보 찾기
            for line in reversed(lines[-50:]):  # 마지막 50줄만 확인
                if 'loss:' in line.lower():
                    return line.strip()
            
            return None
        except Exception as e:
            logger.error(f"로그에서 손실값 추출 실패: {e}")
            return None
    
    async def _extract_error_from_log(self, log_file: str) -> str:
        """로그 파일에서 에러 정보 추출"""
        try:
            if not os.path.exists(log_file):
                return "로그 파일을 찾을 수 없음"
                
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 에러 관련 키워드가 포함된 줄 찾기
            error_keywords = ['error', 'exception', 'failed', 'traceback']
            error_lines = []
            
            for line in reversed(lines[-100:]):  # 마지막 100줄만 확인
                if any(keyword in line.lower() for keyword in error_keywords):
                    error_lines.append(line.strip())
                    if len(error_lines) >= 5:  # 최대 5줄
                        break
            
            return '\n'.join(reversed(error_lines)) if error_lines else "구체적인 에러 정보 없음"
            
        except Exception as e:
            logger.error(f"로그에서 에러 정보 추출 실패: {e}")
            return f"에러 정보 추출 실패: {str(e)}"
    
    async def _send_completion_callback(self, callback_url: str, data: Dict[str, Any]):
        """완료 콜백 전송 (HTTP POST)"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=data, timeout=30) as response:
                    if response.status == 200:
                        logger.info(f"콜백 전송 성공: {callback_url}")
                    else:
                        logger.warning(f"콜백 전송 실패: {response.status}")
                        
        except ImportError:
            logger.warning("aiohttp가 설치되지 않아 콜백을 전송할 수 없습니다")
        except Exception as e:
            logger.error(f"콜백 전송 중 오류: {e}")
    
    async def _upload_to_s3(self, bucket_name: str, file_paths: list, output_name: str):
        """S3에 모델 파일 업로드"""
        try:
            import aioboto3
            
            aws_access_key_id, aws_secret_access_key = self._get_aws_credentials()
            
            session = aioboto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=self.s3_region
            )
            async with session.client('s3') as s3:
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        file_name = os.path.basename(file_path)
                        s3_key = f"models/{output_name}/{file_name}"
                        
                        await s3.upload_file(file_path, bucket_name, s3_key)
                        logger.info(f"S3 업로드 완료: s3://{bucket_name}/{s3_key}")
                        
        except ImportError:
            logger.warning("aioboto3가 설치되지 않아 S3 업로드를 건너뜁니다")
        except Exception as e:
            logger.error(f"S3 업로드 중 오류: {e}")
    
    
    async def _process_message(self, message):
        """
        SQS 메시지 처리
        
        Args:
            message: SQS 메시지
        """
        try:
            # 메시지 본문 파싱
            message_data = json.loads(message['Body'])
            request_id = message_data.get('request_id')
            logger.info(f"메시지 수신: {message_data.get('style_name', 'Unknown')}, request_id: {request_id}")
            
            # 즉시 REQUESTED 상태로 설정
            if request_id:
                self._update_training_status(
                    request_id, 
                    'REQUESTED', 
                    style_name=message_data.get('style_name', 'Unknown'),
                    message_received_at=datetime.utcnow().isoformat() + 'Z'
                )
            
            # 메시지 즉시 삭제
            receipt_handle = message['ReceiptHandle']
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info("메시지 삭제 완료")
            
            # 학습 명령어 구성
            cmd, dirs = self._build_training_command(message_data)
            
            # 비동기 학습 시작
            output_name = message_data.get('style_name', 'unknown_model')
            asyncio.create_task(self._run_training_async(cmd, output_name, message_data, dirs))
            
            logger.info(f"학습 작업 큐에 추가됨: {output_name}")
            
        except json.JSONDecodeError as e:
            traceback.print_exc()
            logger.error(f"메시지 JSON 파싱 오류: {e}")
        except ValueError as e:
            traceback.print_exc()
            logger.error(f"메시지 검증 오류: {e}")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"메시지 처리 중 오류: {e}")
            
            # 메시지 처리 실패 시 상태 업데이트
            request_id = None
            try:
                message_data = json.loads(message['Body'])
                request_id = message_data.get('request_id')
            except:
                pass
            
            if request_id:
                self._update_training_status(
                    request_id, 
                    'FAILED', 
                    error_msg=f"메시지 처리 중 오류: {str(e)}"
                )
    
    async def start_polling(self):
        """SQS 폴링 시작"""
        if not self.sqs:
            self._init_sqs_client()
        
        logger.info(f"SQS 폴링 시작 - 큐: {self.queue_url}")
        
        while True:
            try:
                # SQS에서 메시지 수신
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=self.sqs_max_messages,
                    VisibilityTimeout=self.sqs_visibility_timeout,
                    WaitTimeSeconds=self.sqs_wait_time,  # Long polling
                    MessageAttributeNames=['All']
                )
                
                messages = response.get('Messages', [])
                
                if messages:
                    logger.info(f"{len(messages)}개 메시지 수신")
                    
                    # 각 메시지를 병렬로 처리
                    tasks = [self._process_message(message) for message in messages]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    logger.debug("수신된 메시지 없음")
                
            except ClientError as e:
                logger.error(f"SQS 클라이언트 오류: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"폴링 중 오류 발생: {e}")
                await asyncio.sleep(5)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='AWS SQS LoRA 학습 핸들러')
    parser.add_argument(
        '--queue-url',
        default=SQS_URL_LORA_TRAINING,
        help='SQS 큐 URL'
    )
    parser.add_argument(
        '--region',
        default='ap-northeast-2',
        help='AWS 리전 (기본값: ap-northeast-2)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='로그 레벨 (기본값: INFO)'
    )
    # identifier
    parser.add_argument(
        '--identifier',
        default='shs',
        help='학습 식별자 (기본값: shs)'
    )
    parser.add_argument(
        '--tmp-dir',
        default='./work_dir',
        help='작업 디렉토리 (기본값: ./work_dir)'
    )
    parser.add_argument(
        '--sqs-max-number-of-messages',
        type=int,
        default=10,
        help='SQS에서 한 번에 수신할 최대 메시지 수 (기본값: 10)'
    )
    parser.add_argument(
        '--sqs-visibility-timeout',
        type=int,
        default=3,
        help='SQS 메시지 가시성 타임아웃 (초) (기본값: 3)'
    )
    parser.add_argument(
        '--sqs-wait-time-seconds',
        type=int,
        default=20,
        help='SQS Long Polling 대기 시간 (초) (기본값: 20)'
    )
    parser.add_argument(
        '--s3-bucket-name',
        default='hairmake',
        help='S3 버킷 이름'
    )
    parser.add_argument(
        '--s3-region',
        default='ap-northeast-2',
        help='S3 리전 (기본값: SQS와 동일한 리전 사용)'
    )
    parser.add_argument(
        '--virtual-env-bin-path',
        default='/home/ilseo/source/kohya_ss/venv/bin',
        help='사용할 conda 환경명 (지정하지 않으면 현재 환경 사용)'
    )
    
    args = parser.parse_args()
    
    # 로그 레벨 설정
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 핸들러 생성 및 실행
    handler = LoRATrainingHandler(
        args.queue_url, 
        args.region, 
        args.identifier, 
        args.tmp_dir,
        args.sqs_max_number_of_messages,
        args.sqs_visibility_timeout,
        args.sqs_wait_time_seconds,
        args.s3_bucket_name,
        args.s3_region,
        args.virtual_env_bin_path
    )
    
    try:
        asyncio.run(handler.start_polling())
    except KeyboardInterrupt:
        logger.info("프로그램 종료 요청됨")
        # 실행 중인 프로세스들 정리
        for process in handler.running_processes:
            if process.returncode is None:
                process.terminate()
                logger.info(f"프로세스 종료: PID {process.pid}")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 