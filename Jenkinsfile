
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

        stage('Project Build') {
            steps {
                echo 'Project files are available.'
            }
        }

        stage('Complete') {
            steps {
                echo 'Jenkins Pipeline executed successfully!'
            }
        }
    }
}
