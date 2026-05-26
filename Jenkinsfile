// Jenkinsfile — AlphaGuard CI/CD Pipeline
// Triggers on every push to main and develop branches
// Stages: Checkout → Lint → Test → Build → Deploy → Smoke → Notify

pipeline {
    agent any

    environment {
        IMAGE_NAME    = "alphaguard"
        IMAGE_TAG     = "${env.GIT_COMMIT?.take(8) ?: 'latest'}"
        COMPOSE_FILE  = "docker-compose.yml"
        APP_PORT      = "8000"
        APP_URL       = "http://localhost:${APP_PORT}"
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 30, unit: 'MINUTES')
        timestamps()
    }

    triggers {
        githubPush()
    }

    stages {

        // ── Stage 1: Checkout ───────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
                echo "Branch: ${env.BRANCH_NAME} | Commit: ${env.GIT_COMMIT}"
            }
        }

        // ── Stage 2: Install Dependencies ──────────────────────────────────
        stage('Install') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    pip install -r requirements-dev.txt
                    pip install -e . --no-deps
                '''
            }
        }

        // ── Stage 3: Lint (fail fast on style errors) ───────────────────────
        stage('Lint') {
            parallel {
                stage('flake8') {
                    steps {
                        sh 'flake8 ml/ api/ tests/ --max-line-length=120 --exclude=__pycache__'
                    }
                }
                stage('black') {
                    steps {
                        sh 'black --check ml/ api/ tests/ --line-length=120'
                    }
                }
            }
        }

        // ── Stage 4: Tests + Coverage ───────────────────────────────────────
        stage('Test') {
            steps {
                sh '''
                    pytest tests/ \
                        --cov=ml \
                        --cov=api \
                        --cov-report=xml:coverage.xml \
                        --cov-report=html:htmlcov \
                        --cov-fail-under=70 \
                        -v \
                        --tb=short
                '''
            }
            post {
                always {
                    // Publish test coverage report in Jenkins UI
                    publishHTML(target: [
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report',
                        keepAll: true,
                    ])
                    junit(testResults: 'test-results.xml', allowEmptyResults: true)
                }
            }
        }

        // ── Stage 5: Docker Build ────────────────────────────────────────────
        stage('Build Docker Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                sh '''
                    docker build \
                        --target runtime \
                        --tag ${IMAGE_NAME}:${IMAGE_TAG} \
                        --tag ${IMAGE_NAME}:latest \
                        --cache-from ${IMAGE_NAME}:latest \
                        .
                '''
                echo "Built image: ${IMAGE_NAME}:${IMAGE_TAG}"
            }
        }

        // ── Stage 6: Deploy (only on main) ──────────────────────────────────
        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh '''
                    # Pull latest config and restart containers
                    docker-compose -f ${COMPOSE_FILE} pull prometheus grafana
                    docker-compose -f ${COMPOSE_FILE} up -d app prometheus grafana
                    echo "Deployment started. Waiting for app to be healthy..."
                    sleep 15
                '''
            }
        }

        // ── Stage 7: Smoke Test ──────────────────────────────────────────────
        stage('Smoke Test') {
            when { branch 'main' }
            steps {
                sh '''
                    # Retry health check 5 times with 5s delay
                    for i in 1 2 3 4 5; do
                        STATUS=$(curl -s -o /dev/null -w "%{http_code}" ${APP_URL}/health)
                        if [ "$STATUS" = "200" ]; then
                            echo "✅ Health check passed (attempt $i)"
                            exit 0
                        fi
                        echo "Attempt $i: got HTTP $STATUS, retrying..."
                        sleep 5
                    done
                    echo "❌ Smoke test FAILED after 5 attempts"
                    exit 1
                '''
            }
        }

    }

    post {
        success {
            echo "✅ Pipeline PASSED — ${IMAGE_NAME}:${IMAGE_TAG} deployed."
            // Uncomment to enable Slack notifications:
            // slackSend(color: 'good', message: "✅ AlphaGuard build #${env.BUILD_NUMBER} passed. Branch: ${env.BRANCH_NAME}")
        }
        failure {
            echo "❌ Pipeline FAILED — check logs."
            // slackSend(color: 'danger', message: "❌ AlphaGuard build #${env.BUILD_NUMBER} FAILED. Branch: ${env.BRANCH_NAME}")
        }
        always {
            cleanWs()   // Clean workspace after every build
        }
    }
}
