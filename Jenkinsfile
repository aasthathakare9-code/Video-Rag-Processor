
pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out Multimodal Video Search Engine'
            }
        }

        stage('Environment Check') {
            steps {
                bat '"C:\\Users\\aasth\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" --version'
                bat '"C:\\Users\\aasth\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pip --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                dir('video_rag_processor') {
                    bat '"C:\\Users\\aasth\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pip install opencv-python numpy Pillow ImageHash'
                }
            }
        }

        stage('Run Automated Tests') {
            steps {
                dir('video_rag_processor') {
                    bat '"C:\\Users\\aasth\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m unittest tests.test_video_processor -v'
                }
            }
        }

        stage('Complete') {
            steps {
                echo 'Jenkins Pipeline executed successfully!'
            }
        }
    }
}
