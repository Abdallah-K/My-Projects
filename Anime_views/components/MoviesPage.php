

<link rel="stylesheet" href="views/styles/movies.css"/>

<section id="MoviesPage">
    <div class="moviesbox">
        <div class="moviestitle">
            <h1>Anime Movies</h1>
        </div>
        <div class="moviesinfo">
            <div class="moviesbtn">
                <button id="MoviePrev"><i class="fa-solid fa-arrow-left"></i></button>
            </div>
            <div class="moviesslider">
                
            </div>
            <div class="moviesbtn">
                <button id="MovieNext"><i class="fa-solid fa-arrow-right"></i></button>
            </div>
        </div>


    </div>


</section>

<script>

    $(function(){
        $("#MovieNext").click(()=>{
            let pos = $(".moviesslider").scrollLeft() + 200;
            $('.moviesslider').animate({scrollLeft:pos},200);
        })

        $("#MoviePrev").click(()=>{
            let pos = $(".moviesslider").scrollLeft() - 200;
            $('.moviesslider').animate({scrollLeft:pos},200);
        })



        var moviesout = "";
        $.post("API/PostModel.php",{
            action:"select",
            tablename:"moviesdata",
        },function(data){
            data = $.parseJSON(data);
            $.each(data,(index,movieitem)=>{
                moviesout +=`
                    <div class="movieCard">
                        <img src="views/images/movies/${movieitem.image}">
                    </div>`;
            })
            $(".moviesslider").html(moviesout);
        })

    })

</script>