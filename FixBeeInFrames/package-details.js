
class PackageDetails extends HTMLElement {

    connectedCallback() {
        this.innerHTML = `
            <div class="details full-width">
                <div class="package-title">Package number [${this.id}]</div>
                <div id="canvas-container"></div>
                <div class="width-90">
                    <div class="flex-full-width-center">
                        <button id='move-chart-left' class="button">
                            Move chart to left
                        </button>
                        <button id='move-chart-right' class="button">
                            Move chart to right
                        </button>
                    </div>
                    <select>
                        <option value=10>10</option>
                        <option value=1>1</option>
                        <option value=100>100</option>
                    </select>
                    <div class="flex-full-width-center">
                        <button id='move-point-left' class="button">
                            Move point to left
                        </button>
                        <button id='move-point-right' class="button">
                            Move point to right
                        </button>
                    </div>
                    <div class="flex-full-width-center">
                        <button id="create-point-at-start" class="button">
                            Create point at start
                        </button>
                        <button id="delete-current-point" class="button">
                            Delete current point
                        </button>
                        <button id="previous-point" class="button">
                            Previous point
                        </button>
                        <button id="next-point" class="button">
                            Next point
                        </button>
                    </div>
                    <button id="download-button" class="button full-width">
                        Download data
                    </button>
                    <a id="file-input" hidden></a>
                </div>
            </div>
        `
        this.details = this.querySelector('.details')
        this.fileInput = this.querySelector('#file-input')
        this.downloadData = this.querySelector('#download-button')
        this.moveChartLeftButton = this.querySelector('#move-chart-left')
        this.moveChartRightButton = this.querySelector('#move-chart-right')
        this.movePointLeftButton = this.querySelector('#move-point-left');
        this.movePointRightButton = this.querySelector('#move-point-right');
        this.createPointAtStart = this.querySelector('#create-point-at-start');
        this.speedSelect = this.querySelector('select');
        this.previousPoint = this.querySelector('#previous-point');
        this.nextPoint = this.querySelector('#next-point');
        this.deleteCurrentPoint = this.querySelector('#delete-current-point');

        this.loadOriginalFile()
            .then(() => this.loadResultFile())
            .then(() => this.
            continueInitialization())
            .catch((error) => {
                console.error(`Something went wrong!`)
                console.error(error)
                this.hidden = true;
            })
    }

    loadOriginalFile() {
        return fetch(`/data/data${this.id}.json`)
            .then(response => response.json())
            .then(json => {
                this.originalJson = json;
            })
    }

    loadResultFile() {
        return fetch(`/result/${this.id}_result.json`)
            .then(response => response.json())
            .then(json => {
                this.resultJson = json;
            });
    }

    continueInitialization() {
        this.initializeDownloadButton()
        this.createChart();
        this.initializeButtons();
    }

    initializeButtons() {
        this.moveChartLeftButton.addEventListener('click', () => {
            this.pointMin = Math.max(this.chartMinPoint, this.pointMin - this.chartMovingSpeed)
            this.pointMax = Math.max(this.chartSize, this.pointMax - this.chartMovingSpeed)
            this.updateChart()
        })

        this.moveChartRightButton.addEventListener('click', () => {
            this.pointMin = Math.min(this.chartMaxPoint - this.chartSize, this.pointMin + this.chartMovingSpeed)
            this.pointMax = Math.min(this.chartMaxPoint, this.pointMax + this.chartMovingSpeed)
            this.updateChart()
        })

        this.movePointLeftButton.addEventListener('click', () => {
            this.currentPointIndex = Math.max(this.pointMin + parseInt(this.speedSelect.value),
                this.currentPointIndex - parseInt(this.speedSelect.value));
            this.updateChart()
        })

        this.movePointRightButton.addEventListener('click', () => {
            this.currentPointIndex = Math.min(this.pointMax - parseInt(this.speedSelect.value),
                this.currentPointIndex + parseInt(this.speedSelect.value));
            this.updateChart()
        })

        this.createPointAtStart.addEventListener('click', () => {
            this.resultJson[this.currentPointIndex] = 1
            this.currentPointIndex = this.pointMin + 100;
            this.updateChart()
        })

        this.previousPoint.addEventListener('click', () => {
            for (let i = this.currentPointIndex - 1; i >= 0; i--) {
                if (this.resultJson[i] === 1) {
                    this.resultJson[this.currentPointIndex] = 1;
                    this.currentPointIndex = i
                    this.resultJson[this.currentPointIndex] = 0;
                    this.updateChart()
                    break;
                }
            }
        })

        this.nextPoint.addEventListener('click', () => {
            for (let i = this.currentPointIndex + 1; i < this.resultJson.length; i++) {
                if (this.resultJson[i] === 1) {
                    this.resultJson[this.currentPointIndex] = 1;
                    this.currentPointIndex = i
                    this.resultJson[this.currentPointIndex] = 0;
                    this.updateChart()
                    break;
                }
            }
        })

        this.deleteCurrentPoint.addEventListener('click', () => {
            let newPointIndex = this.currentPointIndex
            for (let i = this.currentPointIndex + 1; i < this.resultJson.length; i++) {
                if (this.resultJson[i] === 1) {
                    newPointIndex = i;
                    break;
                }
            }
            for (let i = this.currentPointIndex - 1; i >= 0; i--) {
                if (this.resultJson[i] === 1) {
                    newPointIndex = i;
                    break;
                }
            }
            if (newPointIndex !== this.currentPointIndex) {
                this.currentPointIndex = newPointIndex;
                this.resultJson[this.currentPointIndex] = 0
                this.updateChart()
            }
        })
    }

    initializeDownloadButton() {
        this.downloadData.addEventListener('click', () => {
            this.details.style.backgroundColor = "var(--success)"
            this.resultJson[this.currentPointIndex] = 1;
            let file = new Blob([JSON.stringify(this.resultJson)], {type: "text/plain"})
            this.fileInput.href = URL.createObjectURL(file)
            this.fileInput.download = `${this.id}_fixed_result.json`
            this.fileInput.click()
        })
    }

    createChart() {
        this.chartMaxPoint = 10000;
        this.chartMinPoint = 0;
        this.chartSize = 2500
        this.pointMin = this.chartMinPoint
        this.pointMax = this.pointMin + this.chartSize
        this.chartMovingSpeed = 500;

        this.updateChart()
    }

    updateChart() {
        if (this.chart !== null && this.chart !== undefined) {
            this.chart.destroy();
            this.querySelector(`#full-chart-${this.id}`).remove();
        }

        const chartObject = document.createElement("canvas")
        chartObject.id = `full-chart-${this.id}`
        chartObject.className = "my-chart";
        this.querySelector("#canvas-container").appendChild(chartObject)

        let chartId = 'full-chart-' + this.id
        this.chart = new Chart(chartId, {
            type: "line",
            data: {
                labels: this.getLabels(),
                datasets: this.getDataSets()
            },
            options: {
                animation: {
                    duration: 0
                }
            }
        });
    }

    getLabels() {
        let labels = []
        for (let i = 0; i <= this.pointMax - this.pointMin; i++) {
            labels[i] = i + this.pointMin
        }
        return labels;
    }

    getDataSets() {
        let currentPointDataset = []
        for (let i = 0; i <= this.resultJson.length; i++) {
            currentPointDataset[i] = 0
            if (this.resultJson[i] === 1 && this.currentPointIndex === undefined) {
                this.currentPointIndex = i
                currentPointDataset[i] = 1
                this.resultJson[i] = 0
            }
        }
        currentPointDataset[this.currentPointIndex] = 1
        return [
            {
                label: 'Currently edited',
                borderColor: "rgba(2,124,2,0.56)",
                fill: false,
                pointRadius: 0,
                data: currentPointDataset.slice(this.pointMin, this.pointMax)
            },
            {
                label: 'Manual data',
                borderColor: "rgba(255,0,0,1)",
                fill: false,
                pointRadius: 0,
                data: this.resultJson.slice(this.pointMin, this.pointMax)
            },
            {
                label: 'Interior',
                borderColor: "rgba(0,0,255,0.5)",
                fill: false,
                pointRadius: 0,
                data: this.originalJson.interior.slice(this.pointMin, this.pointMax)
            },
            {
                label: 'Exterior',
                borderColor: "rgba(255,255,0,0.5)",
                fill: false,
                pointRadius: 0,
                data: this.originalJson.exterior.slice(this.pointMin, this.pointMax)
            },
        ]
    }
}

customElements.define('package-details', PackageDetails);