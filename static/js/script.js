//              Init owlCarousel
$(document).ready(function(){
    $('.plan-carousel').owlCarousel({
        loop: false,
        margin: 10,
        nav: false,
        responsive: {
            0: {
                items: 1
            },
            600: {
                items: 1
            },
            1000: {
                items: 1
            }
        }
    })
});

$(document).ready(function(){
    $('.team-carousel').owlCarousel({
        loop: false,
        margin: 10,
        nav: false,
        dots: false,
        responsive: {
            0: {
                items: 1
            },
            600: {
                items: 1
            },
            800:{
                items:2
            },
            1000: {
                items: 3
            },
            1300:{
                items:4
            }
        }
    })
});

//              GAME
