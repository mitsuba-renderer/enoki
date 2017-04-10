pipeline {
    agent any

    stages {
        stage('Check style') {
            steps {
                ansiColor('xterm') {
                    sh 'resources/check-style.sh'
                }
            }
        }

        stage('Build [debug]') {
            steps {
                sh '''export PATH=$PATH:/usr/local/bin
mkdir -p build-debug
cd build-debug
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
ninja'''
            }
        }

        stage('Test [debug]') {
            steps {
                sh '''export PATH=$PATH:/usr/local/bin
cd build-debug
ctest'''
            }
        }

        stage('Build [release]') {
            steps {
                sh '''export PATH=$PATH:/usr/local/bin
mkdir -p build-release
cd build-release
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
ninja'''
            }
        }

        stage('Test [release]') {
            steps {
                sh '''export PATH=$PATH:/usr/local/bin
cd build-release
ctest'''
            }
        }
    }

    post {
        always {
            deleteDir()
        }

        success {
            slackSend (color: '#00FF00', message: "SUCCESSFUL: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.RUN_DISPLAY_URL})")
        }

        failure {
            slackSend (color: '#FF0000', message: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.RUN_DISPLAY_URL})")

            emailext (
                    subject: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                    body: """<p>FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]':</p>
                        <p>Check console output at &QUOT;<a href='${env.RUN_DISPLAY_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a>&QUOT;</p>""",
                    recipientProviders: [[$class: 'DevelopersRecipientProvider']]
            )
        }
    }
}
