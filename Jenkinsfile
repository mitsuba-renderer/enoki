pipeline {
    agent none

    stages {
        stage ('Check style') {
            agent { label 'knl' }
            steps {
                ansiColor('xterm') {
                    sh 'resources/check-style.sh'
                }
            }
        }
        stage ('Build & Test [debug]') {
            steps {
                parallel (
                    'knl': {
                         node('knl') {
                             sh '''export PATH=$PATH:/usr/local/bin
rm -Rf build-debug
mkdir build-debug
cd build-debug
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
ninja
ctest .'''
                         }
                    },
                    'aarch64': {
                         node('aarch64') {
                             checkout scm
                             sh '''export PATH=$PATH:/usr/local/bin
rm -Rf build-debug
mkdir build-debug
cd build-debug
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
ninja -j3
ctest .'''
                         }
                    }
                )
            }
        }
        stage ('Build & Test [release]') {
            steps {
                parallel (
                    'knl': {
                         node('knl') {
                             sh '''export PATH=$PATH:/usr/local/bin
rm -Rf build-release
mkdir build-release
cd build-release
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
ninja
ctest .'''
                         }
                    },
                    'aarch64': {
                         node('aarch64') {
                             sh '''export PATH=$PATH:/usr/local/bin
rm -Rf build-release
mkdir build-release
cd build-release
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
ninja -j3
ctest .'''
                         }
                    }
                )
            }
        }
    }

    post {
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
