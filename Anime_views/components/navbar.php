

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css" integrity="sha512-xh6O/CkQoPOWDdYTDqeRdPCVd1SpvCA9XXcUnZS2FmJNp1coAFzvtCN9BmamE+4aHK8yyUHUSCcJHgXloTyT2A==" crossorigin="anonymous" referrerpolicy="no-referrer" />

<link rel="stylesheet" href="views/styles/navbar.css"/>

<?php include_once "./config.php" ?>

<nav>
    <div class="navimg">
        <img src="views/images/logo.png">
    </div>
    <div class="navbtn">
        <button id="sideBtn"><i class="fa-solid fa-bars"></i></button>
    </div>
</nav>
<div class="sidenav">
    <div class="sideheader">
        <h1>AnimeWorld</h1>
    </div>
    <div class="sidelinks">
        <a href="index.php">Home</a>
        <a href="discover.php">Discover</a>
        <a href="popular.php">Popular</a>
        <a href="quotesPage.php">Quotes</a>
        <a href="gallery.php">Gallery</a>
    </div>
</div>

<script>

    $(function(){
        $("#sideBtn").click(()=>{
            $(".sidenav").toggleClass("showNav");
        })
    })


</script>