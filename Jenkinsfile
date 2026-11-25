
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
                bat 'python --version'
                bat 'pip --version'
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
