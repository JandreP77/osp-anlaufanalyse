


document.addEventListener("DOMContentLoaded", async function () {
    async function importAds() {
        async function getAds() {
            const requestOptions = {
                method: 'GET',
                redirect: 'follow'
            };

            return await fetch('https://www.leichtathletik.de/wettkaempfe/werbung', requestOptions)
            .then(response => response.text())
            .then(result => {
                return result
            })
            .catch(error => console.log('error', error));
        }

        const data = await getAds();

        const content = document.createElement('div');
        content.innerHTML = data.trim();

        const topBannerContent = document.createElement('div');
        topBannerContent.innerHTML = content.querySelector('.js-top-banner').innerHTML.trim();
        if (topBannerContent.innerHTML !== '') {
            topBannerContent.getElementsByTagName('script')[0].remove();
        }
        document.querySelector('.js-include-top-banner').innerHTML = topBannerContent.innerHTML;

        const footerBannerContent = document.createElement('div');
        footerBannerContent.innerHTML = content.querySelector('.js-footer-banner').innerHTML.trim();
        if (footerBannerContent.innerHTML !== '') {
            footerBannerContent.getElementsByTagName('script')[0].remove();
        }
        document.querySelector('.js-include-footer-banner').innerHTML = footerBannerContent.innerHTML;

        const stickyMobileBannerContent = document.createElement('div');
        stickyMobileBannerContent.innerHTML = content.querySelector('.js-mobile-sticky-banner').innerHTML.trim();
        if (stickyMobileBannerContent.innerHTML !== '') {
            stickyMobileBannerContent.getElementsByTagName('script')[0].remove();
        }
        document.querySelector('.js-include-sticky-mobile-banner').innerHTML = stickyMobileBannerContent.innerHTML;
    }

    function initSliders() {
        const elms = document.getElementsByClassName('js-splide');

        for (let i = 0; i < elms.length; i++) {
            const splide = new Splide(elms[i], {});
            splideGallery(splide, elms[i]);
            splide.mount();
        }
    }

    function initTracking() {
        const ids = ['#splide01-slide01', '#splide01-slide02', '#splide02-slide01', '#splide02-slide02', '#splide03-slide01', '#splide03-slide02'];

        ids.forEach((id) => {
            if (document.querySelector(id)) {
                const dataSet = document.querySelector(id).children[0].dataset
                createObserver(id, dataSet.bannerObject, dataSet.bannerType)
            }
        });
    }

    await importAds();

    initSliders();
    initTracking();
});
