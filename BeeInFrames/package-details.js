
class PackageDetails extends HTMLElement {

    connectedCallback() {
        this.videoLink = `/data/video${this.id}`

        this.innerHTML = `
            <div class="details full-width">
                <div class="package-title">Package number [${this.id}]</div>
                <video id="video" controls>
                    <source src="${this.videoLink}.mp4" type="video/mp4">
                </video>
                <button id="bee-button" class="button full-width">Bzzz! Click if you see a bee!</button>
                <button id="download-button" class="button full-width">
                    Download data
                   
                </button>
                <a id="file-input" hidden></a>
                <button id="reset-button" class="button full-width">Reset data</button>
            </div>
        `
        this.details = this.querySelector('.details')
        this.video = this.querySelector('video')
        this.fileInput = this.querySelector('#file-input')
        this.downloadData = this.querySelector('#download-button')
        this.resetButton = this.querySelector('#reset-button')
        this.beeButton = this.querySelector('#bee-button')

        this.result = []
        this.indexes = []

        this.loadOriginalFile()
            .then(() => this.loadSynchFile())
            .then(() => this.continueInitialization())
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

    loadSynchFile() {
        fetch(`/data/sync${this.id}.json`)
            .then(response => response.json())
            .then(json => {
                this.synchJson = json;
            });
    }

    continueInitialization() {
        for(let i=0; i<this.originalJson.interior.length; i++) {
            this.result[i] = 0;
            this.indexes[i] = i;
        }
        this.initializeResetButton();
        this.initializeDownloadButton();
        this.initializeBeeButton();
    }

    initializeResetButton() {
        this.resetButton.addEventListener('click', () => {
            this.details.style.backgroundColor = "var(--inprogress)"
            for(let i=0; i<this.result.length; i++) {
                this.result[i] = 0;
            }
        })
    }

    initializeDownloadButton() {
        this.downloadData.addEventListener('click', () => {
            this.details.style.backgroundColor = "var(--success)"
            let file = new Blob([JSON.stringify(this.result)], {type: "text/plain"})
            this.fileInput.href = URL.createObjectURL(file)
            this.fileInput.download = `${this.id}_result.json`
            this.fileInput.click()
        })
    }

    initializeBeeButton() {
        this.beeButton.addEventListener('click', () => {
            const timeFrame = Math.floor(this.video.currentTime * 30);
            const resultingTime = this.synchJson[timeFrame];
            this.result[resultingTime] = 1;
        })
    }
}

customElements.define('package-details', PackageDetails);